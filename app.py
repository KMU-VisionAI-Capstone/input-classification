import gradio as gr
from inference import classify
from Image_Captioning.one_image_captioning import captioning


def get_classification_ui(predictions):
    """
    분류 결과 UI를 반환합니다.
    :param predictions: 예측 결과 리스트 ([(class_name, probability), ...])
    :return: HTML 및 그래프 출력
    """

    # 결과 HTML 텍스트 생성
    html_result = "<h2>Top Predictions</h2>"
    for label, score in predictions:
        html_result += f"<p><strong>{label}:</strong> {score:.2f}%</p>"

    return html_result


def get_caption_ui(caption):
    """
    캡션 결과 UI를 반환합니다.
    :param caption: 이미지 캡션 문자열
    :return: HTML 출력
    """

    # 결과 HTML 텍스트 생성
    html_result = f"<h2>Generated Caption</h2><p>{caption}</p>"

    return html_result


def update_classification_ui(image_input):
    predictions = classify(image_input)
    classification_html = get_classification_ui(predictions)
    return classification_html


def update_caption_ui(image_input):
    caption = captioning(image_input)
    caption_html = get_caption_ui(caption)
    return caption_html


def on_click(image_input):
    classification_html = update_classification_ui(image_input)
    caption_html = update_caption_ui(image_input)

    return classification_html, caption_html


# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image()
            predict_button = gr.Button("Run Prediction")
        with gr.Column():
            classification_output = gr.HTML()
            caption_output = gr.HTML()

    predict_button.click(
        fn=on_click,
        inputs=image_input,
        outputs=[classification_output, caption_output],
    )


demo.launch(share=True)
