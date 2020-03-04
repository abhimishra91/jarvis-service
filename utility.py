import pickle
import torch
import re


def prepare_features(seq_1, max_seq_length = 300, zero_pad = False, include_CLS_token = True, include_SEP_token = True):
        
        # Load tokenizer
    with open('./output_artefacts/{TOKENIZER.PKL}', 'rb') as f:
        tokenizer = pickle.load(f)
    
        ## Tokenzine Input
    tokens_a = tokenizer.tokenize(str(seq_1))

        ## Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    
        ## Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    
    ## Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
        ## Input Mask 
    input_mask = [1] * len(input_ids)
    
        ## Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask

    

def preprocessing(string):
    string = str(string)
    string = string.lower()
    string = re.sub("[eE] [Mm]ail|[eE]-[Mm]ail|[eE][Mm]ail", "email", string) 
    legal = "[a-zA-Z]* may contain confidential information *|________________________________ *|This email*|This email contains.*|Mail to:.*|gaat voortdurend na of.*|Antes de imprimir este.*|Facebook:.*|Please ensure you have.*|This message is not intended.*|This email may contain.*|Unless otherwise stated.*|Trading of futures options swaps.*|including any attachments.*|is authorised and regulated.*|This communication is issued by.*|Diese email ent.*|If you received this email in error.*|This message is for information.*|The opinions estimates and.*|Diese Mitteilung ist vertraulich.*|Ce message contient des informations.*|This email, including.*|The information in this email.*|This is an automated email.*|This message and any files.*|Les conversations.*|This email including.*|Follow us.*|we officially launched our new brand.*|This email message is confidential.*|The information contained in this.*|You have received a secure message.*|This email is confidential and subject.*|The power of global connections.*|mailto:.*|Subject:.*|This email and any attach.*|This email is confidential.*|This email \\(including any attachment.*|This email may contain confidential*| =============================.*|If you are not the intended recipient.*|This email is subject.*|Regarding Securities and Insurance.*|Disclaimer.*| The information transmitted.*|Except where specifically stated.*|This communication is for.*|HSBC Bank plc may be solicited.*|This email is intended.*|Unless otherwise stated, this communication.*|This email and any attach.*|This email is for informational purposes only.*|This email is subject to all waivers and other terms.*|This email \\(including any attachment.*|This email, its content and any files transmitted.*|This communication and any attachments are confidential.*|This content of this email and any attachments is confidential.*|The content, including attachments, is a confidential.*|Information in this message is confidential.*|If you have received this communication in error.*|The information contained in this email and any attachments is confidential.*|This message and any attachments.*|This message, and any attachments, is for the.*|This message is subject to terms available at.*|This message is intended only for.*|This message is for information purposes only, it is not a.*|This message may contain confidential.*|This is a system generated communication.*|Ce message et toutes les pieces jointes.*|Message may also be privileged.*|Please consider the environment before.*|Go green. Consider the environment before printing.*|Regards.*|Thanks.*|regards.*|thanks.*|Le informazioni contenute in questo messaggio di posta elettronica sono riservate.*" 
    legal = legal.lower()
    string = re.sub("\\s", " ", string)
    string = re.sub(legal,'', string)
    string = re.sub(r"[^\w\s\.,']", ' ', string)
    string = string.split()
    string = string[:300]

    return (" ".join(string))
