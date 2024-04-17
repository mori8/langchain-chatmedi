import logging
import os
import re

import gradio as gr
from dotenv import load_dotenv

from hugginggpt.history import ConversationHistory
from hugginggpt.llm_factory import create_llms
from hugginggpt.log import setup_logging
from hugginggpt.resources import (
    GENERATED_RESOURCES_DIR,
    get_resource_url,
    init_resource_dirs,
    load_audio,
    load_image,
    save_audio,
    save_image,
)
from main import compute, planning_task_and_selecting_model, infer_model_and_generate_response


load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)
init_resource_dirs()

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")


class Client:
    def __init__(self) -> None:
        self.llms = None
        self.llm_history = ConversationHistory()
        self.last_user_input = ""

    @property
    def is_init(self) -> bool:
        return (
            os.environ.get("OPENAI_API_KEY")
            and os.environ.get("OPENAI_API_KEY").startswith("sk-")
            and os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            and os.environ.get("HUGGINGFACEHUB_API_TOKEN").startswith("hf_")
        )

    def add_text(self, user_input, messages):
        if not self.is_init:
            return (
                "Please set your OpenAI API key and Hugging Face token first!!!",
                messages,
            )
        if not self.llms:
            self.llms = create_llms()

        self.last_user_input = user_input
        try:
            messages = display_message(
                role="user", message=user_input, messages=messages, save_media=True
            )
        except Exception as e:
            logger.exception("")
            error_message = f"Sorry, encountered error: {e}. Please try again. Check logs if problem persists."
            messages = display_message(
                role="assistant",
                message=error_message,
                messages=messages,
                save_media=False,
            )
        return "", messages

    def bot(self, messages):
        if not self.is_init:
            return {}, messages
        try:
            user_input = self.last_user_input
            # TODO: compute 두단계로 분리
            # response, task_summaries = compute(
            #     user_input=user_input,
            #     history=self.llm_history,
            #     llms=self.llms,
            # )

            tasks, hf_models = planning_task_and_selecting_model(
                user_input=user_input,
                history=self.llm_history,
                llms=self.llms,
            )

            messages = display_message(
                role="assistant", message=str(hf_models), messages=messages, save_media=False
            )

            response, task_summaries = infer_model_and_generate_response(
                user_input=user_input,
                tasks=tasks,
                llms=self.llms,
                hf_models=hf_models,
            )

            # TODO: 메세지는 여기서 띄움
            messages = display_message(
                role="assistant", message=response, messages=messages, save_media=False
            )
            self.llm_history.add(role="user", content=user_input)
            self.llm_history.add(role="assistant", content="")

            return task_summaries, messages
        except Exception as e:
            logger.exception("")
            error_message = f"Sorry, encountered error: {e}. Please try again. Check logs if problem persists."
            messages = display_message(
                role="assistant",
                message=error_message,
                messages=messages,
                save_media=False,
            )
            return [], messages


css = ".json {height: 527px; overflow: scroll;} .json-holder {height: 527px; overflow: scroll;}"
with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1><center>ChatMini</center></h1>")
    if not OPENAI_KEY:
        with gr.Row().style():
            with gr.Column(scale=0.85):
                openai_api_key = gr.Textbox(
                    show_label=False,
                    placeholder="Set your OpenAI API key here and press Enter",
                    lines=1,
                    type="password",
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                btn1 = gr.Button("Submit").style(full_height=True)

    if not HUGGINGFACE_TOKEN:
        with gr.Row().style():
            with gr.Column(scale=0.85):
                hugging_face_token = gr.Textbox(
                    show_label=False,
                    placeholder="Set your Hugging Face Token here and press Enter",
                    lines=1,
                    type="password",
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                btn3 = gr.Button("Submit").style(full_height=True)

    with gr.Row().style():
        with gr.Column(scale=0.6):
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)
        with gr.Column(scale=0.4):
            results = gr.JSON(elem_classes="json")

    with gr.Row().style():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter. The url must contain the media type. e.g, https://example.com/example.jpg",
                lines=1,
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn2 = gr.Button("Send").style(full_height=True)

    def set_key(openai_api_key):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return openai_api_key

    def set_token(hugging_face_token):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_face_token
        return hugging_face_token

    def add_text(state, user_input, messages):
        return state["client"].add_text(user_input, messages)

    def bot(state, messages):
        return state["client"].bot(messages)

    if not OPENAI_KEY or not HUGGINGFACE_TOKEN:
        openai_api_key.submit(set_key, [openai_api_key], [openai_api_key])
        btn1.click(set_key, [openai_api_key], [openai_api_key])
        hugging_face_token.submit(set_token, [hugging_face_token], [hugging_face_token])
        btn3.click(set_token, [hugging_face_token], [hugging_face_token])

    state = gr.State(value={"client": Client()})

    txt.submit(add_text, [state, txt, chatbot], [txt, chatbot]).then(
        bot, [state, chatbot], [results, chatbot]
    )
    btn2.click(add_text, [state, txt, chatbot], [txt, chatbot]).then(
        bot, [state, chatbot], [results, chatbot]
    )

    gr.Examples(
        examples=[
            "What are the common symptoms of type 2 diabetes?",
            "A patient has been experiencing chest pain and shortness of breath. Based on these symptoms, what could be the possible diagnosis?",
            "Generate an image of a healthy lung based on the following description: 'A normal chest X-ray showing clear lung fields without any abnormalities.'"
        ],
        inputs=txt,
    )


def display_message(role: str, message: str, messages: list, save_media: bool):
    # Text
    messages.append(format_message(role=role, message=message))

    # Media
    image_urls, audio_urls = extract_medias(message)
    for image_url in image_urls:
        image_url = get_resource_url(image_url)
        if save_media:
            image = load_image(image_url)
            image_url = save_image(image)
            image_url = GENERATED_RESOURCES_DIR + image_url
        messages.append(format_message(role=role, message=(image_url,)))

    for audio_url in audio_urls:
        audio_url = get_resource_url(audio_url)
        if save_media:
            audio = load_audio(audio_url)
            audio_url = save_audio(audio)
            audio_url = GENERATED_RESOURCES_DIR + audio_url
        messages.append(format_message(role=role, message=(audio_url,)))

    return messages


def format_message(role, message):
    if role == "user":
        return message, None
    if role == "assistant":
        return None, message
    else:
        raise ValueError("role must be either user or assistant")


def extract_medias(message: str):
    image_pattern = re.compile(
        r"(http(s?):|\/)?([\.\/_\w:-])*?\.(jpg|jpeg|tiff|gif|png)"
    )
    image_urls = []
    for match in image_pattern.finditer(message):
        if match.group(0) not in image_urls:
            image_urls.append(match.group(0))

    audio_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(flac|wav)")
    audio_urls = []
    for match in audio_pattern.finditer(message):
        if match.group(0) not in audio_urls:
            audio_urls.append(match.group(0))

    return image_urls, audio_urls


demo.launch()
