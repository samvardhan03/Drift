//! ISIN -> NSE/BSE symbol resolution, for broker exports (e.g. Kotak
//! Securities) that identify holdings by ISIN instead of trading symbol.
//!
//! A static cache is consulted first; everything else goes to Yahoo
//! Finance's search endpoint, and every outcome is classified into an
//! `IsinResolutionResult` so callers can tell the user *why* a holding
//! couldn't be analysed (delisted, bond, fund, SME, transient failure).

use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;

pub const YAHOO_SEARCH_URL: &str = "https://query2.finance.yahoo.com/v1/finance/search";

/// ISIN -> Yahoo symbol for well-known stocks, so common uploads skip the
/// network. **Every entry is exactly what Yahoo's own search returns for
/// that ISIN, and the symbol has price history** (checked when this table
/// was built) -- so the cache can never disagree with the live resolver.
/// Do not add pairs from memory: most ISIN/symbol pairs recalled by hand
/// are wrong, and a wrong entry silently swaps one company for another.
static ISIN_CACHE: &[(&str, &str)] = &[
    ("INE742F01042", "ADANIPORTS.NS"),
    ("INE238A01034", "AXISBANK.NS"),
    ("INE296A01024", "BAJFINANCE.NS"),
    ("INE545U01014", "BANDHANBNK.NS"),
    ("INE028A01039", "BANKBARODA.NS"),
    ("INE397D01024", "BHARTIARTL.NS"),
    ("INE752H01013", "CARERATING.NS"),
    ("INE259A01022", "COLPAL.NS"),
    ("INE491A01021", "CUB.NS"),
    ("INE361B01024", "DIVISLAB.NS"),
    ("INE089A01031", "DRREDDY.NS"),
    ("INE129A01019", "GAIL.NS"),
    ("INE860A01027", "HCLTECH.NS"),
    ("INE040A01034", "HDFCBANK.NS"),
    ("INE158A01026", "HEROMOTOCO.NS"),
    ("INE090A01021", "ICICIBANK.NS"),
    ("INE009A01021", "INFY.NS"),
    ("INE154A01025", "ITC.NS"),
    ("INE237A01028", "KOTAKBANK.NS"),
    ("INE018A01030", "LT.NS"),
    ("INE101A01026", "M&M.NS"),
    ("INE585B01010", "MARUTI.NS"),
    ("INE213A01029", "ONGC.NS"),
    ("INE002A01018", "RELIANCE.NS"),
    ("INE148I01020", "SAMMAANCAP.NS"),
    ("INE062A01020", "SBIN.NS"),
    ("INE070A01015", "SHREECEM.NS"),
    ("INE721A01013", "SHRIRAMFIN.NS"),
    ("INE671H01015", "SOBHA.NS"),
    ("INE044A01036", "SUNPHARMA.NS"),
    ("INE192A01025", "TATACONSUM.NS"),
    ("INE245A01021", "TATAPOWER.NS"),
    ("INE081A01020", "TATASTEEL.NS"),
    ("INE467B01029", "TCS.NS"),
    ("INE669C01036", "TECHM.NS"),
    ("INE280A01028", "TITAN.NS"),
    ("INE694A01020", "UNITECH.NS"),
    ("INE528G01027", "YESBANK.NS"),
];

/// The cached symbol for `isin`, if any.
pub fn cached_symbol(isin: &str) -> Option<&'static str> {
    ISIN_CACHE.iter().find(|(i, _)| *i == isin).map(|(_, s)| *s)
}

/// Outcome of resolving one ISIN. String payloads: the symbol for
/// `Resolved`; the instrument kind for `NonEquity` ("Mutual fund",
/// "Bond/Debenture", "Currency/SGB"); the ISIN or a short detail for the
/// rest.
#[derive(Debug, Clone, PartialEq)]
pub enum IsinResolutionResult {
    Resolved(String),
    /// Reserved for a confirmed delisting. Yahoo's search gives no delisting
    /// signal, so the resolver itself never returns this today (an unknown
    /// ISIN is `NotFound`); it exists so callers/messages are complete.
    Delisted(String),
    NonEquity(String),
    SmeListed(String),
    NotFound(String),
    Timeout(String),
    RateLimited(String),
}

/// Tunables, overridable so tests run in milliseconds against a stub.
#[derive(Debug, Clone)]
pub struct ResolverConfig {
    pub search_url: String,
    pub first_timeout: Duration,
    /// Timeout of the single retry after a timeout.
    pub retry_timeout: Duration,
    /// Wait before the single retry after a 429.
    pub rate_limit_wait: Duration,
    pub batch_size: usize,
    /// Sleep between batches.
    pub batch_gap: Duration,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        ResolverConfig {
            search_url: YAHOO_SEARCH_URL.to_string(),
            first_timeout: Duration::from_secs(5),
            retry_timeout: Duration::from_secs(8),
            rate_limit_wait: Duration::from_secs(2),
            batch_size: 5,
            batch_gap: Duration::from_millis(200),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    #[serde(default)]
    quotes: Vec<Quote>,
}

#[derive(Debug, Deserialize)]
struct Quote {
    #[serde(default)]
    symbol: String,
    #[serde(default)]
    exchange: String,
    #[serde(default, rename = "quoteType")]
    quote_type: String,
}

enum Fetched {
    Quotes(Vec<Quote>),
    Timeout,
    RateLimited,
    /// Connection error, 5xx, or an unparseable body.
    Failed,
}

async fn fetch_once(
    cfg: &ResolverConfig,
    isin: &str,
    client: &reqwest::Client,
    (lang, count): (&str, &str),
    timeout: Duration,
) -> Fetched {
    let sent = client
        .get(&cfg.search_url)
        .query(&[("q", isin), ("lang", lang), ("region", "IN"), ("quotesCount", count), ("newsCount", "0")])
        .timeout(timeout)
        .send()
        .await;
    let resp = match sent {
        Ok(r) => r,
        Err(e) if e.is_timeout() => return Fetched::Timeout,
        Err(_) => return Fetched::Failed,
    };
    if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Fetched::RateLimited;
    }
    if !resp.status().is_success() {
        return Fetched::Failed;
    }
    match resp.json::<SearchResponse>().await {
        Ok(body) => Fetched::Quotes(body.quotes),
        Err(e) if e.is_timeout() => Fetched::Timeout,
        Err(_) => Fetched::Failed,
    }
}

/// One search with the retry policy: a timeout (or other transient failure)
/// is retried once with the longer timeout; a 429 is retried once after
/// `rate_limit_wait`.
async fn fetch(cfg: &ResolverConfig, isin: &str, client: &reqwest::Client, locale: (&str, &str)) -> Fetched {
    match fetch_once(cfg, isin, client, locale, cfg.first_timeout).await {
        Fetched::Timeout | Fetched::Failed => fetch_once(cfg, isin, client, locale, cfg.retry_timeout).await,
        Fetched::RateLimited => {
            tokio::time::sleep(cfg.rate_limit_wait).await;
            fetch_once(cfg, isin, client, locale, cfg.first_timeout).await
        }
        ok => ok,
    }
}

fn is_nse_or_bse(exchange: &str) -> bool {
    matches!(exchange, "NSI" | "NSE" | "BSI" | "BSE" | "BOM")
}

/// NSE/BSE SME-platform listings carry an `-SM`/`-SME` series suffix.
fn is_sme(symbol: &str) -> bool {
    let base = symbol.rsplit_once('.').map_or(symbol, |(b, _)| b).to_uppercase();
    base.ends_with("-SME") || base.ends_with("-SM")
}

/// Decides from a search response: tradeable NSE/BSE equity or ETF (`.NS`
/// preferred over `.BO`) -> `Resolved`; else an SME listing, a fund, a
/// currency or a bond quote -> the matching skip; `None` if nothing
/// decisive (so the caller can retry another locale or fall back).
fn classify(quotes: &[Quote]) -> Option<IsinResolutionResult> {
    let tradeable = |q: &&Quote, suffix: &str| {
        matches!(q.quote_type.as_str(), "EQUITY" | "ETF")
            && is_nse_or_bse(&q.exchange)
            && q.symbol.ends_with(suffix)
            && !is_sme(&q.symbol)
    };
    if let Some(q) = quotes.iter().find(|q| tradeable(q, ".NS")).or_else(|| quotes.iter().find(|q| tradeable(q, ".BO"))) {
        return Some(IsinResolutionResult::Resolved(q.symbol.clone()));
    }
    if let Some(q) = quotes.iter().find(|q| is_sme(&q.symbol)) {
        return Some(IsinResolutionResult::SmeListed(q.symbol.clone()));
    }
    for q in quotes {
        match q.quote_type.as_str() {
            "MUTUALFUND" => return Some(IsinResolutionResult::NonEquity("Mutual fund".to_string())),
            "CURRENCY" => return Some(IsinResolutionResult::NonEquity("Currency/SGB".to_string())),
            "BOND" => return Some(IsinResolutionResult::NonEquity("Bond/Debenture".to_string())),
            _ => {}
        }
    }
    None
}

/// Last resort when Yahoo returns nothing: Indian ISINs encode the
/// instrument class, so some non-equities can still be named. `INE`
/// issuers use security-type digits at positions 8-9 (`01` = equity share,
/// `07` = debenture); `IN0`-`IN3` prefixes are government securities
/// (including Sovereign Gold Bonds). Anything else -- including `INF`
/// funds, which may be an ETF Yahoo simply doesn't know -- is `NotFound`.
fn classify_by_isin_structure(isin: &str) -> Option<IsinResolutionResult> {
    let b = isin.as_bytes();
    if b.len() != 12 {
        return None;
    }
    if isin.starts_with("INF") {
        return Some(IsinResolutionResult::NonEquity("Mutual fund".to_string()));
    }
    if isin.starts_with("INE") && &isin[7..9] == "07" {
        return Some(IsinResolutionResult::NonEquity("Bond/Debenture".to_string()));
    }
    if matches!(b[2], b'0'..=b'3') {
        return Some(IsinResolutionResult::NonEquity("Currency/SGB".to_string()));
    }
    None
}

/// Resolves one ISIN with the default config (see `resolve_isin_with`).
pub async fn resolve_isin_to_ticker(isin: &str, client: &reqwest::Client) -> IsinResolutionResult {
    resolve_isin_with(&ResolverConfig::default(), isin, client).await
}

/// Cache -> Yahoo search (`en-US`, 5 quotes) -> if nothing decisive, a
/// second search (`en-IN`, 10 quotes) -> ISIN-structure fallback ->
/// `NotFound`. Timeouts and 429s are retried per `fetch`; if the retry also
/// fails the result is `Timeout`/`RateLimited`.
pub async fn resolve_isin_with(cfg: &ResolverConfig, isin: &str, client: &reqwest::Client) -> IsinResolutionResult {
    if let Some(symbol) = cached_symbol(isin) {
        return IsinResolutionResult::Resolved(symbol.to_string());
    }
    for locale in [("en-US", "5"), ("en-IN", "10")] {
        match fetch(cfg, isin, client, locale).await {
            Fetched::Quotes(quotes) => {
                if let Some(result) = classify(&quotes) {
                    return result;
                }
            }
            Fetched::Timeout => return IsinResolutionResult::Timeout(isin.to_string()),
            Fetched::RateLimited => return IsinResolutionResult::RateLimited(isin.to_string()),
            Fetched::Failed => return IsinResolutionResult::Timeout(format!("{isin} (market data service error)")),
        }
    }
    classify_by_isin_structure(isin).unwrap_or_else(|| IsinResolutionResult::NotFound(isin.to_string()))
}

/// Why a holding was left out of the analysis -- user-facing.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct IsinSkipReason {
    pub isin: String,
    pub reason: String,
    /// "delisted" | "non_equity" | "sme" | "not_found" | "timeout"
    pub category: String,
}

#[derive(Debug, Clone, Default)]
pub struct IsinBatchResult {
    /// `(isin, symbol, value_inr)`, in input order.
    pub resolved: Vec<(String, String, f64)>,
    pub skipped: Vec<IsinSkipReason>,
}

/// The `(category, user message)` for a non-`Resolved` result.
pub fn skip_reason(isin: &str, result: &IsinResolutionResult) -> Option<IsinSkipReason> {
    let (category, reason) = match result {
        IsinResolutionResult::Resolved(_) => return None,
        IsinResolutionResult::Delisted(_) => (
            "delisted",
            format!(
                "This stock appears to be delisted or suspended. It was removed from your portfolio analysis. ({isin})"
            ),
        ),
        IsinResolutionResult::NonEquity(kind) if kind == "Mutual fund" => (
            "non_equity",
            "Mutual fund units cannot be analysed with equity risk models. Remove this from your portfolio or \
             replace it with the underlying equity ETF equivalent."
                .to_string(),
        ),
        IsinResolutionResult::NonEquity(kind) if kind.starts_with("Bond") => (
            "non_equity",
            format!("Bonds and debentures are fixed-income instruments and are not supported. Skipped: {isin}."),
        ),
        IsinResolutionResult::NonEquity(_) => (
            "non_equity",
            format!("Sovereign Gold Bonds and currency instruments are not supported. Skipped: {isin}."),
        ),
        IsinResolutionResult::SmeListed(_) => (
            "sme",
            format!("This stock is listed on the SME platform and has insufficient market data for risk analysis. Skipped: {isin}."),
        ),
        IsinResolutionResult::NotFound(_) => (
            "not_found",
            format!(
                "Could not identify this security. It may be a recently listed stock or use a non-standard identifier. Skipped: {isin}."
            ),
        ),
        IsinResolutionResult::Timeout(_) | IsinResolutionResult::RateLimited(_) => (
            "timeout",
            format!("Market data service timed out while looking up {isin}. Please try again."),
        ),
    };
    Some(IsinSkipReason { isin: isin.to_string(), reason, category: category.to_string() })
}

/// Resolves `(isin, value_inr)` pairs. Cache hits are answered immediately;
/// the rest go to Yahoo in batches of `batch_size`, run concurrently within
/// a batch, with `batch_gap` between batches. A repeated ISIN is looked up
/// once. Output follows input order.
pub async fn resolve_isins_batch(isins: &[(String, f64)], client: &reqwest::Client) -> IsinBatchResult {
    resolve_isins_batch_with(&ResolverConfig::default(), isins, client).await
}

pub async fn resolve_isins_batch_with(
    cfg: &ResolverConfig,
    isins: &[(String, f64)],
    client: &reqwest::Client,
) -> IsinBatchResult {
    let mut outcome: HashMap<String, IsinResolutionResult> = HashMap::new();
    let mut pending: Vec<String> = Vec::new();
    for (isin, _) in isins {
        if outcome.contains_key(isin) || pending.contains(isin) {
            continue;
        }
        match cached_symbol(isin) {
            Some(symbol) => {
                outcome.insert(isin.clone(), IsinResolutionResult::Resolved(symbol.to_string()));
            }
            None => pending.push(isin.clone()),
        }
    }

    for (n, batch) in pending.chunks(cfg.batch_size.max(1)).enumerate() {
        if n > 0 {
            tokio::time::sleep(cfg.batch_gap).await;
        }
        let mut set = tokio::task::JoinSet::new();
        for isin in batch {
            let (cfg, client, isin) = (cfg.clone(), client.clone(), isin.clone());
            set.spawn(async move {
                let result = resolve_isin_with(&cfg, &isin, &client).await;
                (isin, result)
            });
        }
        while let Some(joined) = set.join_next().await {
            // A task only fails by panicking (a bug); treat it as a transient
            // failure for that ISIN rather than taking the whole upload down.
            if let Ok((isin, result)) = joined {
                outcome.insert(isin, result);
            }
        }
    }

    let mut out = IsinBatchResult::default();
    let mut skipped_seen: Vec<&String> = Vec::new();
    for (isin, value) in isins {
        let result = outcome
            .get(isin)
            .cloned()
            .unwrap_or_else(|| IsinResolutionResult::Timeout(isin.clone()));
        match &result {
            IsinResolutionResult::Resolved(symbol) => out.resolved.push((isin.clone(), symbol.clone(), *value)),
            other => {
                if !skipped_seen.contains(&isin) {
                    skipped_seen.push(isin);
                    out.skipped.extend(skip_reason(isin, other));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    #[derive(Clone, Debug)]
    struct Seen {
        isin: String,
        lang: String,
        count: String,
        at: Instant,
    }

    struct Reply {
        delay: Duration,
        status: u16,
        body: String,
    }

    fn reply(status: u16, body: String) -> Reply {
        Reply { delay: Duration::ZERO, status, body }
    }

    /// Stub Yahoo search: `handler(n, seen)` (n = how many requests for this
    /// ISIN so far, 0-based) decides the reply. One thread per connection,
    /// so a deliberately slow reply doesn't block other requests.
    fn stub(handler: impl Fn(usize, &Seen) -> Reply + Send + Sync + 'static) -> (String, Arc<Mutex<Vec<Seen>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/finance/search", listener.local_addr().unwrap());
        let log: Arc<Mutex<Vec<Seen>>> = Arc::new(Mutex::new(Vec::new()));
        let (log2, handler) = (log.clone(), Arc::new(handler));
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { return };
                let (log, handler) = (log2.clone(), handler.clone());
                std::thread::spawn(move || {
                    let mut buf = [0u8; 4096];
                    let n = conn.read(&mut buf).unwrap_or(0);
                    let head = String::from_utf8_lossy(&buf[..n]).to_string();
                    let query = head.split(' ').nth(1).and_then(|p| p.split('?').nth(1)).unwrap_or("");
                    let param = |k: &str| {
                        query.split('&').find_map(|kv| kv.strip_prefix(&format!("{k}="))).unwrap_or("").to_string()
                    };
                    let seen = Seen { isin: param("q"), lang: param("lang"), count: param("quotesCount"), at: Instant::now() };
                    let nth = {
                        let mut l = log.lock().unwrap();
                        let nth = l.iter().filter(|s| s.isin == seen.isin).count();
                        l.push(seen.clone());
                        nth
                    };
                    let r = handler(nth, &seen);
                    std::thread::sleep(r.delay);
                    let _ = write!(
                        conn,
                        "HTTP/1.1 {} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        r.status,
                        r.body.len(),
                        r.body
                    );
                });
            }
        });
        (url, log)
    }

    fn quotes_body(items: &[(&str, &str, &str)]) -> String {
        let q: Vec<String> = items
            .iter()
            .map(|(s, e, t)| format!(r#"{{"symbol":"{s}","exchange":"{e}","quoteType":"{t}"}}"#))
            .collect();
        format!(r#"{{"quotes":[{}]}}"#, q.join(","))
    }

    fn fast_cfg(url: String) -> ResolverConfig {
        ResolverConfig {
            search_url: url,
            first_timeout: Duration::from_millis(150),
            retry_timeout: Duration::from_millis(600),
            rate_limit_wait: Duration::from_millis(250),
            batch_size: 5,
            batch_gap: Duration::from_millis(200),
        }
    }

    async fn resolve(url: String, isin: &str) -> IsinResolutionResult {
        resolve_isin_with(&fast_cfg(url), isin, &reqwest::Client::new()).await
    }

    #[tokio::test]
    async fn cache_hit_returns_without_any_http_call() {
        let (url, log) = stub(|_, _| reply(200, quotes_body(&[])));
        assert_eq!(resolve(url, "INE002A01018").await, IsinResolutionResult::Resolved("RELIANCE.NS".into()));
        assert!(log.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn equity_and_etf_quotes_resolve_preferring_ns() {
        let (url, _) = stub(|_, _| reply(200, quotes_body(&[("ABC.BO", "BSE", "EQUITY"), ("ABC.NS", "NSI", "EQUITY")])));
        assert_eq!(resolve(url, "INE000000001").await, IsinResolutionResult::Resolved("ABC.NS".into()));
        let (url, _) = stub(|_, _| reply(200, quotes_body(&[("GOLDBEES.NS", "NSI", "ETF")])));
        assert_eq!(resolve(url, "INF000000001").await, IsinResolutionResult::Resolved("GOLDBEES.NS".into()));
        let (url, _) = stub(|_, _| reply(200, quotes_body(&[("ONLY.BO", "BSE", "EQUITY")])));
        assert_eq!(resolve(url, "INE000000002").await, IsinResolutionResult::Resolved("ONLY.BO".into()));
    }

    #[tokio::test]
    async fn an_inf_isin_that_yahoo_lists_as_an_etf_still_resolves() {
        let (url, _) = stub(|_, _| reply(200, quotes_body(&[("METALIETF.NS", "NSI", "EQUITY")])));
        assert_eq!(resolve(url, "INF109KC19W1").await, IsinResolutionResult::Resolved("METALIETF.NS".into()));
    }

    #[tokio::test]
    async fn bond_fund_and_currency_quotes_are_non_equity() {
        for (kind, label) in [("BOND", "Bond/Debenture"), ("MUTUALFUND", "Mutual fund"), ("CURRENCY", "Currency/SGB")] {
            let (url, _) = stub(move |_, _| reply(200, quotes_body(&[("X123", "NMS", kind)])));
            assert_eq!(resolve(url, "INE000000003").await, IsinResolutionResult::NonEquity(label.into()), "{kind}");
        }
    }

    #[tokio::test]
    async fn sme_listing_is_sme_listed_not_resolved() {
        let (url, _) = stub(|_, _| reply(200, quotes_body(&[("TINYCO-SM.NS", "NSI", "EQUITY")])));
        assert_eq!(resolve(url, "INE000000004").await, IsinResolutionResult::SmeListed("TINYCO-SM.NS".into()));
    }

    #[tokio::test]
    async fn empty_quotes_retry_with_the_other_locale_then_not_found() {
        let (url, log) = stub(|_, _| reply(200, quotes_body(&[])));
        assert_eq!(resolve(url, "INE999X99999").await, IsinResolutionResult::NotFound("INE999X99999".into()));
        let seen = log.lock().unwrap().clone();
        assert_eq!(seen.len(), 2);
        assert_eq!((seen[0].lang.as_str(), seen[0].count.as_str()), ("en-US", "5"));
        assert_eq!((seen[1].lang.as_str(), seen[1].count.as_str()), ("en-IN", "10"));
    }

    #[tokio::test]
    async fn the_second_locale_can_rescue_an_empty_first_response() {
        let (url, _) = stub(|_, seen| {
            if seen.lang == "en-IN" {
                reply(200, quotes_body(&[("RESCUED.NS", "NSI", "EQUITY")]))
            } else {
                reply(200, quotes_body(&[]))
            }
        });
        assert_eq!(resolve(url, "INE000000005").await, IsinResolutionResult::Resolved("RESCUED.NS".into()));
    }

    #[tokio::test]
    async fn a_timeout_is_retried_once_with_the_longer_timeout() {
        let (url, log) = stub(|nth, _| {
            if nth == 0 {
                Reply { delay: Duration::from_millis(400), status: 200, body: quotes_body(&[]) } // > 150ms first timeout
            } else {
                reply(200, quotes_body(&[("SLOW.NS", "NSI", "EQUITY")]))
            }
        });
        assert_eq!(resolve(url, "INE000000006").await, IsinResolutionResult::Resolved("SLOW.NS".into()));
        assert_eq!(log.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn a_persistent_timeout_is_reported_as_timeout() {
        let (url, log) = stub(|_, _| Reply { delay: Duration::from_millis(1500), status: 200, body: quotes_body(&[]) });
        assert!(matches!(resolve(url, "INE000000007").await, IsinResolutionResult::Timeout(_)));
        assert_eq!(log.lock().unwrap().len(), 2, "one try plus one retry");
    }

    #[tokio::test]
    async fn a_429_waits_then_retries_once() {
        let (url, log) = stub(|nth, _| {
            if nth == 0 {
                reply(429, "{}".into())
            } else {
                reply(200, quotes_body(&[("OK.NS", "NSI", "EQUITY")]))
            }
        });
        assert_eq!(resolve(url, "INE000000008").await, IsinResolutionResult::Resolved("OK.NS".into()));
        let seen = log.lock().unwrap().clone();
        assert!(seen[1].at.duration_since(seen[0].at) >= Duration::from_millis(250));

        let (url, log) = stub(|_, _| reply(429, "{}".into()));
        assert!(matches!(resolve(url, "INE000000009").await, IsinResolutionResult::RateLimited(_)));
        assert_eq!(log.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn isin_structure_names_bonds_and_government_securities_when_yahoo_has_nothing() {
        let (url, _) = stub(|_, _| reply(200, quotes_body(&[])));
        // INE + type digits "07" = debenture; IN0 prefix = government security.
        assert_eq!(resolve(url.clone(), "INE342T07379").await, IsinResolutionResult::NonEquity("Bond/Debenture".into()));
        assert_eq!(resolve(url.clone(), "IN0020220011").await, IsinResolutionResult::NonEquity("Currency/SGB".into()));
        // An unknown equity stays NotFound; an unmatched INF code is a mutual fund.
        assert!(matches!(resolve(url.clone(), "INE2WKE01011").await, IsinResolutionResult::NotFound(_)));
        assert_eq!(resolve(url, "INF179KC1DI2").await, IsinResolutionResult::NonEquity("Mutual fund".into()));
    }

    #[tokio::test]
    async fn a_batch_of_six_runs_as_five_then_one_with_a_gap() {
        let (url, log) = stub(|_, seen| reply(200, quotes_body(&[(&format!("S{}.NS", &seen.isin[8..]), "NSI", "EQUITY")])));
        let isins: Vec<(String, f64)> = (1..=6).map(|n| (format!("INE00000{n:04}"), n as f64)).collect();
        let r = resolve_isins_batch_with(&fast_cfg(url), &isins, &reqwest::Client::new()).await;
        assert_eq!(r.resolved.len(), 6);
        assert!(r.skipped.is_empty());
        // Order and value pass-through.
        assert_eq!(r.resolved[0].0, "INE000000001");
        assert_eq!(r.resolved[5].2, 6.0);

        let seen = log.lock().unwrap().clone();
        assert_eq!(seen.len(), 6);
        let sixth = seen.iter().find(|s| s.isin == "INE000000006").unwrap().at;
        let last_of_first_batch = seen.iter().filter(|s| s.isin != "INE000000006").map(|s| s.at).max().unwrap();
        assert!(sixth.duration_since(last_of_first_batch) >= Duration::from_millis(190), "gap between batches");
    }

    #[tokio::test]
    async fn batch_mixes_cache_hits_skips_and_duplicates() {
        let (url, log) = stub(|_, seen| match seen.isin.as_str() {
            "INE000000011" => reply(200, quotes_body(&[("AAA.NS", "NSI", "EQUITY")])),
            "INF000000012" => reply(200, quotes_body(&[("0P0001.BO", "BSE", "MUTUALFUND")])),
            _ => reply(200, quotes_body(&[])),
        });
        let isins: Vec<(String, f64)> = vec![
            ("INE002A01018".into(), 100.0), // cached
            ("INE000000011".into(), 50.0),
            ("INF000000012".into(), 25.0),
            ("INE000000011".into(), 10.0), // duplicate row
            ("INE000000013".into(), 5.0),
        ];
        let r = resolve_isins_batch_with(&fast_cfg(url), &isins, &reqwest::Client::new()).await;
        let resolved: Vec<_> = r.resolved.iter().map(|(i, s, v)| (i.as_str(), s.as_str(), *v)).collect();
        assert_eq!(
            resolved,
            vec![("INE002A01018", "RELIANCE.NS", 100.0), ("INE000000011", "AAA.NS", 50.0), ("INE000000011", "AAA.NS", 10.0)]
        );
        let cats: Vec<_> = r.skipped.iter().map(|s| (s.isin.as_str(), s.category.as_str())).collect();
        assert_eq!(cats, vec![("INF000000012", "non_equity"), ("INE000000013", "not_found")]);
        // The duplicate was looked up once: 11 (1) + 12 (1) + 13 (2 locales).
        assert_eq!(log.lock().unwrap().len(), 4);
    }

    #[test]
    fn skip_messages_match_the_spec_wording() {
        let msg = |r: IsinResolutionResult| skip_reason("INE1", &r).unwrap();
        assert_eq!(
            msg(IsinResolutionResult::NonEquity("Mutual fund".into())).reason,
            "Mutual fund units cannot be analysed with equity risk models. Remove this from your portfolio or \
             replace it with the underlying equity ETF equivalent."
        );
        assert_eq!(
            msg(IsinResolutionResult::NonEquity("Bond/Debenture".into())).reason,
            "Bonds and debentures are fixed-income instruments and are not supported. Skipped: INE1."
        );
        assert_eq!(
            msg(IsinResolutionResult::NonEquity("Currency/SGB".into())).reason,
            "Sovereign Gold Bonds and currency instruments are not supported. Skipped: INE1."
        );
        assert_eq!(msg(IsinResolutionResult::SmeListed("X".into())).category, "sme");
        assert_eq!(msg(IsinResolutionResult::NotFound("X".into())).category, "not_found");
        assert_eq!(msg(IsinResolutionResult::Delisted("X".into())).category, "delisted");
        assert_eq!(msg(IsinResolutionResult::RateLimited("X".into())).category, "timeout");
        assert!(skip_reason("INE1", &IsinResolutionResult::Resolved("A.NS".into())).is_none());
    }

    #[test]
    fn cache_is_well_formed() {
        let mut seen = std::collections::HashSet::new();
        for (isin, symbol) in ISIN_CACHE {
            assert!(crate::ticker_map::is_isin(isin), "{isin}");
            assert!(symbol.ends_with(".NS"), "{symbol}");
            assert!(seen.insert(*isin), "duplicate {isin}");
        }
    }
}
