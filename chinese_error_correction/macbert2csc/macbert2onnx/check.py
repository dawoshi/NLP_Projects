import onnx
import torch
import numpy as np
import operator
import onnxruntime
import onnxruntime as ort
from transformers import BertTokenizer, BertForMaskedLM


input_names = ["input_ids", "attention_mask", "token_type_ids"]
vocab_size = 10000
batch_size = 1
sequence_length = 160
def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
            # add unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if i >= len(corrected_text):
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details

def create_onnxruntime_input(vocab_size, batch_size, sequence_length, input_names, data_type=np.int64):
    input_ids = np.random.randint(low=0, high=vocab_size - 1, size=(batch_size, sequence_length), dtype=data_type)
    print(input_ids.shape)
    print(input_ids)
    inputs = {"input_ids": input_ids}

    if "attention_mask" in input_names:
        attention_mask = np.ones([batch_size, sequence_length], dtype=data_type)
        inputs["attention_mask"] = attention_mask

    if "token_type_ids" in input_names:
        segment_ids = np.zeros([batch_size, sequence_length], dtype=data_type)
        inputs["token_type_ids"] = segment_ids
    return inputs



def onnx_test():
    model = "/root/autodl-tmp/pre_trained_model/macbert4csc-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model)
    sent = '下个星期，我跟我朋唷打算去法国玩儿'
    tokenized_tokens = tokenizer(sent)
    input_ids = np.array([tokenized_tokens['input_ids']], dtype=np.int64)
    attention_mask = np.array([tokenized_tokens['attention_mask']], dtype=np.int64)
    token_type_ids = np.array([tokenized_tokens['token_type_ids']], dtype=np.int64)



    onnx_model = "./macbert4csc.onnx"

    # 1. print graph

    # Load the ONNX model
    model = onnx.load(onnx_model)

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 15

    # 2. check input and output
    if torch.cuda.is_available():
         ort_session = ort.InferenceSession(onnx_model,
                        sess_options=sess_options,
                        providers=["CUDAExecutionProvider"])
    else:
         ort_session = ort.InferenceSession(onnx_model,
                        sess_options=sess_options,
                        providers = ["CPUExecutionProvider"])
    result = ort_session.run(
        output_names=None,
        input_feed={"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids}
    )[0]
    # print(result)
    for idx, token_id in enumerate(result[0]):
        print(token_id)
        predicted_token = tokenizer.convert_ids_to_tokens([token_id])[0]
        if(token_id == 102):
            source = "SEP"
        else:
            source = sent[idx-1]
        # print('text : {} -> new text : {}'.format(source, predicted_token))
onnx_test()
