import gradio as gr

from src.inference import predict


def classify(text):
    result = predict(text)
    return {
        "negative": result["scores"][0],
        "neutral": result["scores"][1],
        "positive": result["scores"][2],
    }


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        label="Text",
        placeholder="Type a social media post...",
    ),
    outputs=gr.Label(label="Sentiment"),
    title="Twitter Sentiment Analysis",
    description=(
        "Fine-tuned cardiffnlp/twitter-roberta-base-sentiment-latest "
        "on tweet_eval sentiment."
    ),
)


if __name__ == "__main__":
    demo.launch()
