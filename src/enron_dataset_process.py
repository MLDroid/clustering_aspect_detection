import os,nltk,logging
import email
import ast
from nltk.corpus import stopwords
from time import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Flag = False

def get_files_to_process(dirname, extn):
    files_to_process = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)]
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith(extn):
                files_to_process.append(os.path.join(root, f))

    files_to_process = list(set(files_to_process))
    files_to_process = sorted(files_to_process, key=lambda x: int(x.split('/')[4].split('.txt')[0]))
    return files_to_process


stop_string = '-----Original Message-----'
def preprocess_email(filename):
    f = open(filename)
    msg = email.message_from_file(f)
    body = ''

    if msg.is_multipart():
        for payload in msg.get_payload():
            body = body + (payload.get_payload())
            print f, 'this email is multipart'
    else:
        body = (msg.get_payload())

    body = body.split(stop_string)[0].lower().replace('\n', '').strip()
    return body


if Flag:
    output_f = open('enron_parsed_text.txt', 'w')
stopwords = set(stopwords.words('english'))
def preprocess_email_tok_tag(filename):
    email_body = preprocess_email(filename)
    tokens = nltk.word_tokenize(email_body)
    stopped_tokens = []
    for w in tokens:
        if w not in stopwords and '=' not in w:
            stopped_tokens.append(w)
    parsed_text = nltk.pos_tag(stopped_tokens)
    output_f.write('{}\n'.format(parsed_text))
    return parsed_text


def get_parsed_email_bodys(Email_dir, extn):
    t0 = time()
    if Flag:
        Email_files_to_process = get_files_to_process(Email_dir, extn)
        parsed_email_bodys = []
        for f in Email_files_to_process:
            parsed_email_bodys.append(preprocess_email_tok_tag(f))
        output_f.close()
        logger.info('loaded parsed email bodys in {} sec.'.format(round(time() - t0, 2)))
    else:
        parsed_email_bodys = open('enron_parsed_text.txt', 'r').readlines()
        parsed_email_bodys = [ast.literal_eval(parsed_body) for parsed_body in parsed_email_bodys]
        logger.info('loaded parsed email bodys in {} sec.'.format(round(time() - t0, 2)))
    return parsed_email_bodys



if __name__ == '__main__':
    pass


