from Models import model_rnet
import numpy as np
import tensorflow as tf
import argparse
import random
from pprint import pprint

import gensim
from gensim.models import word2vec,Word2Vec
import logging
import random
import os
import json
import csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# np.set_printoptions(threshold=np.inf)

def load_data(context_path='./Preprocess_data/data/train_context.npy',question_path='./Preprocess_data/data/train_question.npy'):
	return np.load(context_path),np.load(question_path)

def load_test(context_path='./Preprocess_data/data/test_context.npy',question_path='./Preprocess_data/data/test_question.npy'):
	return np.load(context_path),np.load(question_path)

def get_batch(batch_no,modOpts):
    si = (batch_no * modOpts['batch_size'])
    ei = min(len(train_context), si + modOpts['batch_size'])
    n = ei - si
    
    tensor_dict = {}
    tensor_dict['paragraph'] = train_context_emb[si:ei]
    tensor_dict['question'] = train_question_emb[si:ei]
    tensor_dict['answer_si'] = answer_si[si:ei]
    tensor_dict['answer_ei'] = answer_ei[si:ei]
    
    return tensor_dict

def get_dict_wordvec(train_context,max_length,n_comp,model_path='model_300_w3'):
	def index_array(X,max_length,n_comp):
		return np.concatenate([[word2idx.get('_PAD') if word2idx.get(x) is None else word2idx.get(x) for x in X],np.zeros((max_length-len(X)))])
	model = Word2Vec.load(model_path)
	word2idx = {"_PAD": 0} 
	vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
	embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
	for i in range(len(vocab_list)):
		word = vocab_list[i][0]
		word2idx[word] = i + 1
		embeddings_matrix[i + 1] = vocab_list[i][1]
	train_context_vec = [index_array(x,max_length,n_comp) for x in train_context] 
	return np.array(train_context_vec),embeddings_matrix

def word_embedding(data,mincount=5,n=250,load=True):
	if load:
		model = Word2Vec.load('model_100_w3.bin')
	else:
		model = Word2Vec(data,sg=1, size=n, window=5, min_count=mincount,workers=8)
		model.save('./model/word/skipgram_')

	print(model.most_similar(['台灣']))
	print(model.most_similar(['中國']))
	print(model.most_similar(['持續']))
    
def f1_score(prediction, ground_truth):
	from collections import Counter

	prediction_tokens = prediction
	ground_truth_tokens = ground_truth
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

def set_word_length_list(context):
    word_length_list = []
    #print(context[0][0])
    #print(context.shape)
    for paragraph in context:
        #print(paragraph)
        #print('-')
        word_length_list.append([len(i) for i in paragraph])
        #print(word_length_list)
    return(word_length_list)

def word_to_character(word_Length,wordID, questionID):
	wordID_start = wordID[0]
	wordID_end = wordID[1]
	characterID_start = sum(word_Length[questionID][0:int(wordID_start)])
	characterID_end = characterID_start+sum(word_Length[questionID][int(wordID_start):int(wordID_end+1)])-1
	return([characterID_start, characterID_end])

def writecsv(test_id,start_end):
	ans = []
	for i in start_end:
		str_=''
		for j in range(i[0],i[1]+1):
			str_ = str_ + str(j) + ' '
		ans.append(str_)
	final_ans=[]
	for i in range(len(ans)):
		final_ans.append([str(test_id[i])])
		final_ans[i].append(ans[i])

	filename = "prediction.csv"
	#filename = args[0][3]

	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","answer"])
	for i in range(len(final_ans)):
		s.writerow(final_ans[i])
	text.close()


test_id_path ='./Preprocess_data/data/test.question.id'
test_id = [line.strip() for line in open(test_id_path,"r", encoding='utf-8-sig')]

#train_context,train_question = load_data()
test_context,test_question = load_test()

n_comp = 300
length_context = 400
length_question = 45

w2vmodel = 'Models/save/model_300_w3'

for i in range(len(test_context)):
	test_context[i] = test_context[i][0:length_context]
	test_question[i] = test_question[i][0:length_question]

test_context_vec, embeddings_matrix = get_dict_wordvec(test_context,length_context,n_comp,w2vmodel)#'./model/model_50.bin'
test_question_vec, embeddings_matrix = get_dict_wordvec(test_question,length_question,n_comp,w2vmodel)


test_context_emb = []
j = 0
for i in range(len(test_context_vec)):
    test_context_emb.append([])
    for j in range(len(test_context_vec[i])):
        test_context_emb[i].append(embeddings_matrix[int(test_context_vec[i][j])])

test_question_emb = []
j = 0
for i in range(len(test_question_vec)):
    test_question_emb.append([])
    for j in range(len(test_question_vec[i])):
        test_question_emb[i].append(embeddings_matrix[int(test_question_vec[i][j])])


modOpts = json.load(open('Models/config.json','r'))['rnet']['dev']

num_samples = len(test_context)
num_batches = int(np.floor(num_samples/modOpts['batch_size']))

model = model_rnet.R_NET(modOpts)
input_tensors, loss, acc, pred_si, pred_ei = model.build_model()
saved_model = 'Models/save/rnet_model_14.ckpt'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

new_saver = tf.train.Saver()
sess = tf.InteractiveSession(config=config)
new_saver.restore(sess, saved_model)

tensor_dict = {}
tensor_dict['paragraph'] = test_context_emb
tensor_dict['question'] = test_question_emb

empty_answer_idx = np.ndarray((len(test_context), length_context))
predictions_for_si = []
predictions_for_ei = []
feed_dict={
			input_tensors['p']:tensor_dict['paragraph'],
			input_tensors['q']:tensor_dict['question'],
			input_tensors['a_si']:empty_answer_idx,
			input_tensors['a_ei']:empty_answer_idx,
			}
predictions_si, predictions_ei = sess.run([pred_si, pred_ei], feed_dict=feed_dict)

ans_si = np.round(predictions_si)
ans_ei = np.round(predictions_ei)

test_word_span = []
for i in range(len(ans_si)):
	test_word_span.append([ans_si[i]])
	test_word_span[i].append(ans_ei[i])

word_Length = set_word_length_list(test_context)
an = []
for i in range(len(test_context)):
    an.append(word_to_character(word_Length,test_word_span[i],i))

writecsv(test_id,an)