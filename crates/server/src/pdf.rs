//! `GET /report/{id}`: a single-page A4 PDF summary of a stored
//! `/experiment` or `/ask` result, built directly on `printpdf` 0.12's
//! `Op`-list API (see `Cargo.toml` -- pinned exactly). That API is a full
//! rewrite from the `PdfLayerReference`-based API most printpdf
//! examples/tutorials still show; this module only uses the low-level `Op`
//! primitives (text positioning, lines), not printpdf's optional HTML/CSS
//! layout engine (which is what pulls in the crate's much heavier
//! dependency tree -- azul-core, azul-layout, hyphenation,
//! rust-fontconfig -- none of which this module touches).
//!
//! Renders from the pieces `store::RiskSnapshot` persists, not the full
//! `/ask` pipeline result directly: `trace` (the `EvidenceTrace`, always
//! present), plus `narration`/`grounding_warnings`, which are `Some`/
//! non-empty only for a `/ask`-originated snapshot (`/experiment` never
//! calls Gemini, so it has none to store).

use compute::format::format_inr;
use compute::trace::EvidenceTrace;
use printpdf::*;
use serde_json::Value;

const PAGE_WIDTH_MM: f32 = 210.0;
const PAGE_HEIGHT_MM: f32 = 297.0;
const LEFT_MARGIN_MM: f32 = 20.0;
const RIGHT_MARGIN_MM: f32 = 20.0;
const CONTENT_WIDTH_MM: f32 = PAGE_WIDTH_MM - LEFT_MARGIN_MM - RIGHT_MARGIN_MM; // 170mm, per spec
const RIGHT_EDGE_MM: f32 = PAGE_WIDTH_MM - RIGHT_MARGIN_MM;

/// Renders `trace` as a one-page PDF and returns the raw file bytes.
/// `narration` is the stored snapshot's narration text (`None` for a
/// `/experiment`-originated snapshot); `grounding_warnings` is its
/// grounding-warning list (empty for `/experiment`, and often empty for
/// `/ask` too -- only non-empty when the narration still had unmatched
/// numbers after retries).
pub fn render_report(trace: &EvidenceTrace, narration: Option<&str>, grounding_warnings: &[String]) -> Vec<u8> {
    let mut doc = PdfDocument::new("Drift Risk Copilot Report");
    let mut page = Page::new();

    let outputs = &trace.outputs;
    let result_value = outputs.get("result");

    // --- Section 1: header ---
    page.text(LEFT_MARGIN_MM, page.y, "Drift Risk Copilot", BuiltinFont::HelveticaBold, 18.0);
    let generated_at = chrono::Utc::now().format("%Y-%m-%d %H:%M UTC");
    let header_right = format!("{} | generated {}", trace.experiment, generated_at);
    page.text_right(RIGHT_EDGE_MM, page.y, &header_right, BuiltinFont::Helvetica, 10.0);
    page.advance(7.0);
    page.rule();
    page.advance(8.0);

    // --- Section 2: narration ---
    page.text(LEFT_MARGIN_MM, page.y, "Analysis", BuiltinFont::HelveticaBold, 12.0);
    page.advance(6.0);
    match narration {
        Some(text) => {
            page.wrapped_text(text, BuiltinFont::Helvetica, 10.0, 5.0);
            if !grounding_warnings.is_empty() {
                page.advance(2.0);
                page.text(
                    LEFT_MARGIN_MM,
                    page.y,
                    "[!] Some numbers in this explanation could not be verified.",
                    BuiltinFont::HelveticaOblique,
                    9.0,
                );
                page.advance(5.0);
            }
        }
        None => {
            page.text(
                LEFT_MARGIN_MM,
                page.y,
                "No narrative \u{2014} direct experiment result.",
                BuiltinFont::HelveticaOblique,
                10.0,
            );
            page.advance(5.0);
        }
    }
    page.advance(4.0);

    // --- Section 3: key numbers ---
    page.text(LEFT_MARGIN_MM, page.y, "Key Numbers", BuiltinFont::HelveticaBold, 12.0);
    page.advance(6.0);
    let rows = key_numbers(&trace.experiment, result_value, &trace.model_params);
    page.table(&["Metric", "Value", "Unit"], &rows);
    page.advance(4.0);

    // --- Section 4: evidence trace (abridged) ---
    page.text(LEFT_MARGIN_MM, page.y, "Evidence Trace", BuiltinFont::HelveticaBold, 12.0);
    page.advance(6.0);

    page.text(
        LEFT_MARGIN_MM,
        page.y,
        &format!(
            "Model: window={} periods, frequency={:?}, shrinkage_intensity={:.4}{}",
            trace.model_params.window_periods,
            trace.model_params.frequency,
            trace.model_params.shrinkage_intensity,
            trace
                .model_params
                .regime_state
                .as_ref()
                .map(|r| format!(", regime={}", r.current_label))
                .unwrap_or_default(),
        ),
        BuiltinFont::Helvetica,
        9.0,
    );
    page.advance(5.0);

    let total_forward_filled: usize =
        trace.data_quality.per_series.iter().map(|s| s.forward_filled_days).sum();
    let total_dropped: usize = trace.data_quality.per_series.iter().map(|s| s.dropped_days).sum();
    page.text(
        LEFT_MARGIN_MM,
        page.y,
        &format!(
            "Data quality: {} to {} ({} trading days), {} forward-filled, {} dropped",
            trace.data_quality.date_range_start,
            trace.data_quality.date_range_end,
            trace.data_quality.trading_days,
            total_forward_filled,
            total_dropped,
        ),
        BuiltinFont::Helvetica,
        9.0,
    );
    page.advance(6.0);

    let invariant_rows: Vec<[String; 3]> = trace
        .invariants
        .iter()
        .map(|inv| {
            [
                inv.name.clone(),
                if inv.passed { "\u{2713}".to_string() } else { "\u{2717}".to_string() },
                inv.detail.clone(),
            ]
        })
        .collect();
    if !invariant_rows.is_empty() {
        page.table(&["Invariant", "Passed", "Detail"], &invariant_rows);
    }

    // --- Section 5: footer ---
    let footer_y = 15.0;
    page.set_y(footer_y + 5.0);
    page.rule();
    page.set_y(footer_y);
    page.text(LEFT_MARGIN_MM, page.y, "Generated by Drift Risk Copilot", BuiltinFont::Helvetica, 9.0);
    page.text_right(
        RIGHT_EDGE_MM,
        page.y,
        "Full-history smoothed regime \u{2014} not suitable for live trading signals.",
        BuiltinFont::Helvetica,
        9.0,
    );

    let pdf_page = PdfPage::new(Mm(PAGE_WIDTH_MM), Mm(PAGE_HEIGHT_MM), page.into_ops());
    doc.with_pages(vec![pdf_page]);
    doc.save(&PdfSaveOptions::default(), &mut Vec::new())
}

/// Experiment-type-specific rows for the "Key Numbers" table, read
/// directly from the trace's `outputs.result` JSON (the same JSON the
/// HTTP API already serves), not from the strongly-typed
/// `*Output` structs -- `PipelineResult` only carries the serialized
/// `EvidenceTrace`, not the original typed output.
fn key_numbers(
    experiment: &str,
    result: Option<&Value>,
    model_params: &compute::trace::ModelParams,
) -> Vec<[String; 3]> {
    let regime_row = || {
        [
            "Current regime".to_string(),
            model_params
                .regime_state
                .as_ref()
                .map(|r| r.current_label.to_string())
                .unwrap_or_else(|| "not requested".to_string()),
            "\u{2014}".to_string(),
        ]
    };

    let Some(result) = result else {
        return vec![];
    };
    let f = |path: &[&str]| -> Option<f64> {
        let mut v = result;
        for key in path {
            v = v.get(key)?;
        }
        v.as_f64()
    };
    let pct = |v: f64| format!("{:.2}", v * 100.0);

    match experiment {
        "FactorShock" => {
            let mut rows = vec![[
                "Portfolio P&L".to_string(),
                format_inr(f(&["portfolio_pnl_inr"]).unwrap_or(0.0)),
                "\u{20b9}".to_string(),
            ]];
            if let Some(Value::Object(given)) = result.get("given_shocks") {
                for (factor, shock) in given {
                    if let Some(simple) = shock.get("simple").and_then(Value::as_f64) {
                        rows.push([
                            format!("Given shock: {factor}"),
                            format!("{:.2}", simple * 100.0),
                            "%".to_string(),
                        ]);
                    }
                }
            }
            if let Some(Value::Object(implied)) = result.get("implied_shocks") {
                for (factor, shock) in implied {
                    if let Some(simple) = shock.get("simple").and_then(Value::as_f64) {
                        rows.push([
                            format!("Implied shock: {factor} (model-estimated)"),
                            format!("{:.2}", simple * 100.0),
                            "%".to_string(),
                        ]);
                    }
                }
            }
            rows.push(regime_row());
            if let Some(crisis_pnl) = f(&["crisis_comparison", "portfolio_pnl_inr"]) {
                rows.push([
                    "Crisis-regime P&L".to_string(),
                    format_inr(crisis_pnl),
                    "\u{20b9}".to_string(),
                ]);
            }
            rows
        }
        "RiskDecomposition" => {
            let mut rows = vec![[
                "Portfolio Vol (annualised)".to_string(),
                pct(f(&["portfolio_vol_annualized"]).unwrap_or(0.0)),
                "%".to_string(),
            ]];
            if let Some(Value::Array(by_factor)) = result.get("by_factor") {
                let mut sorted: Vec<&Value> = by_factor.iter().collect();
                sorted.sort_by(|a, b| {
                    let ca = a.get("fraction_of_vol").and_then(Value::as_f64).unwrap_or(0.0).abs();
                    let cb = b.get("fraction_of_vol").and_then(Value::as_f64).unwrap_or(0.0).abs();
                    cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
                });
                let labels = ["Top factor by Euler contribution", "Second factor"];
                for (label, factor) in labels.iter().zip(sorted.iter()) {
                    let name = factor.get("factor").and_then(Value::as_str).unwrap_or("?");
                    let share = factor.get("fraction_of_vol").and_then(Value::as_f64).unwrap_or(0.0);
                    rows.push([format!("{label}: {name}"), pct(share), "%".to_string()]);
                }
            }
            rows.push([
                "Specific risk share".to_string(),
                pct(f(&["specific_risk_fraction_of_vol"]).unwrap_or(0.0)),
                "%".to_string(),
            ]);
            rows.push(regime_row());
            rows
        }
        "CvarRebalance" => {
            let before = f(&["stats_before", "historical_cvar"]).unwrap_or(0.0);
            let after = f(&["stats_after", "historical_cvar"]);
            let mut rows = vec![["CVaR Before".to_string(), pct(before), "%".to_string()]];
            if let Some(after) = after {
                rows.push(["CVaR After".to_string(), pct(after), "%".to_string()]);
                rows.push(["CVaR Reduction".to_string(), pct(before - after), "%".to_string()]);
            }
            if let Some(turnover) = f(&["turnover"]) {
                rows.push(["Turnover Used".to_string(), pct(turnover), "%".to_string()]);
            }
            if let Some(commission) = f(&["commission_cost_inr"]) {
                rows.push([
                    "Commission Cost".to_string(),
                    format_inr(commission),
                    "\u{20b9}".to_string(),
                ]);
            }
            rows
        }
        "PortfolioPerformance" => {
            let mut rows = vec![
                [
                    "Total Return".to_string(),
                    pct(f(&["total_return"]).unwrap_or(0.0)),
                    "%".to_string(),
                ],
                [
                    "Annualized Return".to_string(),
                    pct(f(&["annualized_return"]).unwrap_or(0.0)),
                    "%".to_string(),
                ],
                [
                    "Annualized Vol (realized)".to_string(),
                    pct(f(&["annualized_vol_realized"]).unwrap_or(0.0)),
                    "%".to_string(),
                ],
                [
                    "Max Drawdown".to_string(),
                    pct(f(&["max_drawdown"]).unwrap_or(0.0)),
                    "%".to_string(),
                ],
            ];
            if let Some(end_value) = f(&["end_value_inr"]) {
                rows.push(["End Value".to_string(), format_inr(end_value), "\u{20b9}".to_string()]);
            }
            rows
        }
        _ => vec![],
    }
}

/// printpdf's built-in fonts are `WinAnsiEncoding` (essentially CP1252, a
/// single-byte encoding) -- confirmed by inspecting a rendered PDF's raw
/// content stream: any character outside that repertoire (the rupee sign,
/// the Unicode minus `compute::format_inr` uses for negative amounts,
/// checkmark/cross, the warning triangle) is silently written as a literal
/// `?` glyph, not an error. Embedding a real Unicode font was ruled out
/// (the checkpoint spec explicitly wants "no font files to embed or
/// manage"), so this substitutes ASCII equivalents for PDF output only --
/// `compute::format_inr` itself, and every other consumer of it (the trace
/// field, JSON responses), keeps using ₹/− unchanged.
fn pdf_safe(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{20b9}' => "Rs.".to_string(),
            '\u{2212}' => "-".to_string(),
            '\u{2713}' => "Y".to_string(),
            '\u{2717}' => "N".to_string(),
            '\u{26a0}' => "!".to_string(),
            other => other.to_string(),
        })
        .collect()
}

fn black() -> Color {
    Color::Rgb(Rgb {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        icc_profile: None,
    })
}

/// Accumulates `Op`s for one page, tracking a "current y" cursor (mm from
/// the bottom, per printpdf's coordinate system) that callers advance
/// top-to-bottom.
struct Page {
    ops: Vec<Op>,
    y: f32,
}

impl Page {
    fn new() -> Self {
        let mut ops = vec![Op::SetFillColor { col: black() }, Op::SetOutlineColor { col: black() }];
        ops.push(Op::SetOutlineThickness { pt: Pt(1.0) });
        Page {
            ops,
            y: PAGE_HEIGHT_MM - 20.0,
        }
    }

    fn into_ops(self) -> Vec<Op> {
        self.ops
    }

    fn advance(&mut self, mm: f32) {
        self.y -= mm;
    }

    fn set_y(&mut self, y_mm: f32) {
        self.y = y_mm;
    }

    fn text(&mut self, x_mm: f32, y_mm: f32, text: &str, font: BuiltinFont, size_pt: f32) {
        let text = pdf_safe(text);
        self.ops.push(Op::StartTextSection);
        self.ops.push(Op::SetTextCursor {
            pos: Point::new(Mm(x_mm), Mm(y_mm)),
        });
        self.ops.push(Op::SetFont {
            font: PdfFontHandle::Builtin(font),
            size: Pt(size_pt),
        });
        self.ops.push(Op::SetLineHeight { lh: Pt(size_pt) });
        self.ops.push(Op::ShowText {
            items: vec![TextItem::Text(text.to_string())],
        });
        self.ops.push(Op::EndTextSection);
    }

    fn text_right(&mut self, right_edge_mm: f32, y_mm: f32, text: &str, font: BuiltinFont, size_pt: f32) {
        let text = pdf_safe(text);
        let width_mm = text_width_mm(&text, font, size_pt);
        self.text(right_edge_mm - width_mm, y_mm, &text, font, size_pt);
    }

    fn rule(&mut self) {
        let y = self.y;
        self.ops.push(Op::DrawLine {
            line: Line {
                points: vec![
                    LinePoint {
                        p: Point::new(Mm(LEFT_MARGIN_MM), Mm(y)),
                        bezier: false,
                    },
                    LinePoint {
                        p: Point::new(Mm(RIGHT_EDGE_MM), Mm(y)),
                        bezier: false,
                    },
                ],
                is_closed: false,
            },
        });
    }

    /// Word-wraps `text` at `CONTENT_WIDTH_MM` and draws it, advancing `y`
    /// by `line_height_mm` per line.
    fn wrapped_text(&mut self, text: &str, font: BuiltinFont, size_pt: f32, line_height_mm: f32) {
        let text = pdf_safe(text);
        for line in wrap_text(&text, font, size_pt, CONTENT_WIDTH_MM) {
            self.text(LEFT_MARGIN_MM, self.y, &line, font, size_pt);
            self.advance(line_height_mm);
        }
    }

    /// A simple 3-column table: header row bold, one row per entry,
    /// columns at fixed fractions of `CONTENT_WIDTH_MM` (55% / 25% / 20%).
    fn table(&mut self, headers: &[&str; 3], rows: &[[String; 3]]) {
        let col_x = [
            LEFT_MARGIN_MM,
            LEFT_MARGIN_MM + CONTENT_WIDTH_MM * 0.55,
            LEFT_MARGIN_MM + CONTENT_WIDTH_MM * 0.80,
        ];
        for (i, h) in headers.iter().enumerate() {
            self.text(col_x[i], self.y, h, BuiltinFont::HelveticaBold, 9.0);
        }
        self.advance(5.0);
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                self.text(col_x[i], self.y, cell, BuiltinFont::Helvetica, 9.0);
            }
            self.advance(4.5);
        }
    }
}

/// Approximate per-character advance width as a fraction of em, loosely
/// modeled on Helvetica's standard AFM metrics (narrow punctuation/`i`/`l`,
/// wide `m`/`w`/uppercase, ~0.52em average for everything else). printpdf's
/// builtin fonts expose no width/measurement API (only external
/// `ParsedFont`s carry `glyph_widths`), so this is a heuristic, not a
/// font-metrics-exact measurement -- see the README's PDF section for the
/// implication (wrap points may drift a little from a real Helvetica
/// renderer's, though not enough to visibly break the 170mm content width
/// in practice for this report's short, mostly-ASCII text).
fn char_width_em(c: char, bold: bool) -> f32 {
    let w: f32 = match c {
        ' ' | '.' | ',' | '\'' | '!' | ':' | ';' | 'i' | 'j' | 'l' | 'I' | '|' => 0.28,
        'f' | 't' | 'r' => 0.33,
        'm' | 'w' | 'M' | 'W' | '%' => 0.83,
        '0'..='9' => 0.556,
        c if c.is_ascii_uppercase() => 0.70,
        _ => 0.52,
    };
    if bold {
        w * 1.05
    } else {
        w
    }
}

fn text_width_mm(text: &str, font: BuiltinFont, size_pt: f32) -> f32 {
    let bold = matches!(
        font,
        BuiltinFont::HelveticaBold | BuiltinFont::TimesBold | BuiltinFont::CourierBold
    );
    let em_width: f32 = text.chars().map(|c| char_width_em(c, bold)).sum();
    let width_pt = em_width * size_pt;
    width_pt * (25.4 / 72.0)
}

fn wrap_text(text: &str, font: BuiltinFont, size_pt: f32, max_width_mm: f32) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{current} {word}")
        };
        if text_width_mm(&candidate, font, size_pt) > max_width_mm && !current.is_empty() {
            lines.push(current);
            current = word.to_string();
        } else {
            current = candidate;
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

