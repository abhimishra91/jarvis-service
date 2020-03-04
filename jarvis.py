from transformers import DistilBertForSequenceClassification
import json
import torch
import pickle
from utility import *

class Jarvis:

    def predict(self, input):
        prediction = list()
        # Load lables
        with open('./output_artefacts/{LABLE_TO_IX.JSON}') as json_file:
            label_to_ix = json.load(json_file)
        
        # Load config
        with open('./output_artefacts/{CONFIG.PKL}', 'rb') as f:
            config = pickle.load(f)
        config.num_labels = len(list(label_to_ix.values()))

        model = DistilBertForSequenceClassification(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = model.cuda()
        model_path = './output_artefacts/{MODEL.PTH}'
        model.load_state_dict(torch.load(model_path, map_location=device))

        msg = preprocessing(str(input))
        model.eval()
        input_msg, _ = prepare_features(msg)
        if torch.cuda.is_available():
            input_msg = input_msg.cuda()
        output = model(input_msg)[0]
        output_exp = torch.exp(output)
        probability = torch.div(output_exp, torch.add(output_exp,1.))
        _, pred_label = probability.topk(1)
        percent = torch.mul(_,100)
        percent = torch.reshape(percent,(1,1)).tolist()
        percent = [item for sublist in percent for item in sublist ]
        for i in pred_label[0]:
            queue=list(label_to_ix.keys())[i]
            prediction.append(queue)
        my_result = dict(zip(prediction,percent))
        
        return my_result

