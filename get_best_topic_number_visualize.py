'''
Estimate best fit number of topics for reason for recall.
Uses tom_lib version 0.2.2
http://mediamining.univ-lyon2.fr/people/guille/tom.php
'''

from tom_lib.structure import corpus
from tom_lib.nlp import topic_model
from tom_lib.visualization import visualization
import psycopg2
import pandas as pd

#Connect to database
conn = psycopg2.connect(database="RecallsReviews2", user="unsafefoods2", password="Password1", host="unsafefoods2.csya4zsfb6y4.us-east-1.rds.amazonaws.com", port="5432")

print("Opened database successfully")

cur = conn.cursor()

#get reason for recall from db
cur.execute('SELECT REASON, EVENT_ID FROM EVENT;')
event_data = pd.DataFrame(cur.fetchall())
reason_text = list(event_data.iloc[:,0])

#create corpus from reason for recall text
corpus = Corpus(text = reason_text,
                language='english', 
                vectorization='tfidf', 
                n_gram=1,
                max_relative_frequency=0.8, 
                min_absolute_frequency=4)
print('corpus size:', corpus.size)
print('vocabulary size:', len(corpus.vocabulary))
print('Vector representation of document 0:\n', corpus.vector_for_document(0))

#create nmf topic model
topic_model = NonNegativeMatrixFactorization(corpus)
viz = Visualization(topic_model)
viz.plot_greene_metric(min_num_topics=5, 
                       max_num_topics=50, 
                       tao=10, step=1, 
                       top_n_words=10)
viz.plot_arun_metric(min_num_topics=5, 
                     max_num_topics=50, 
                     iterations=10)
viz.plot_brunet_metric(min_num_topics=5, 
                       max_num_topics=50,
                       iterations=10)
					   
#save output
utils.save_topic_model(topic_model, 'output/NMF_15topics.tom')
topic_model = utils.load_topic_model('output/NMF_15topics.tom')