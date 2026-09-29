mod support;

use agent::suggest::suggest_follow_up;
use support::{text_response, MockGeminiClient};

#[tokio::test]
async fn suggest_returns_the_mocks_text_response_as_is() {
    let client = MockGeminiClient::new(vec![text_response(
        "What if I cut my turnover budget to 20% instead?",
    )]);

    let suggestion = suggest_follow_up(&client, "Vol is 15.5% annualised.").await.unwrap();

    assert_eq!(suggestion, "What if I cut my turnover budget to 20% instead?");
}

#[tokio::test]
async fn empty_text_response_produces_an_empty_suggestion_without_erroring() {
    let client = MockGeminiClient::new(vec![text_response("")]);

    let suggestion = suggest_follow_up(&client, "Vol is 15.5% annualised.").await.unwrap();

    assert_eq!(suggestion, "");
}
