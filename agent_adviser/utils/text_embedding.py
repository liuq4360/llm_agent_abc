import sys
sys.path.append('./')
from agent_adviser.configs.model_config import MODEL_PATH
from FlagEmbedding import BGEM3FlagModel


MODEL_LIST = MODEL_PATH["embed_model"]


def embed_text(
        text_list: str,
        embed_model: str = 'bge-m3',
):

    model = BGEM3FlagModel(MODEL_LIST[embed_model],
                           use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight
    # performance degradation

    embeddings = model.encode(text_list,
                              batch_size=12,
                              max_length=1024,
                              # If you don't need such a long length, you can set a smaller value to speed up
                              # the encoding process.
                              )['dense_vecs']

    return embeddings.tolist()


if __name__ == '__main__':
    vec = embed_text("What is BGE M3?")
    print(vec)
