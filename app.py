import gradio as gr
from inference import inference


def fancy_ui(predictions):
    """
    팬시한 결과 UI를 반환합니다.
    :param predictions: 예측 결과 리스트 ([(class_name, probability), ...])
    :return: HTML 및 그래프 출력
    """

    # 결과 HTML 텍스트 생성
    html_result = "<h2>Top Predictions</h2>"
    for label, score in predictions:
        html_result += f"<p><strong>{label}:</strong> {score:.2f}%</p>"

    return html_result

# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image()
            predict_button = gr.Button("Run Prediction")
        html_output = gr.HTML()

    def update_ui(image_input):
        predictions = inference(image_input)
        html = fancy_ui(predictions)
        return html

    predict_button.click(update_ui, inputs=image_input, outputs=html_output)


demo.launch(share=True)