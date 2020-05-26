#Import packages

import pandas as pd
from faker import Faker
import numpy as np

#Generate Synthetic data

fake = Faker()
n = 300

word_list = [
'Looking','fun','play',
'for','happy','corona',
'covid','go','lucky',
'out','take','I','you','will' ]

ethinicity = ['Asian', 'Caucasian', 'African-American','South-Asian']
status = ['Single','Married']
interests = ['Sports','Music','News','Current Affairs']
orientation = ['Straight','Gay','Asexual','Bisexual']
word = ['Yes','No']
religion = ['Christianity','Muslim','Budhist','Hindu']

fake_data = pd.DataFrame([[fake.name(),
            np.random.randint(19,91),
            np.random.randint(19,91),            
            np.random.choice(['M.', 'F.']),
            np.random.choice(['M.', 'F.']), 
            fake.address(),
            fake.address(),
            fake.job(),
            fake.job(),
            fake.words(1, ethinicity, True),
            fake.words(1, ethinicity, True),
            fake.words(1,status,True),
            fake.words(1,status,True),
            fake.words(1,interests,True),
            fake.words(1,interests,True),
            fake.words(1,orientation,True),
            fake.words(1,orientation,True),
            fake.words(1,word,True), 
            fake.words(1,word,True), 
            fake.words(1,religion,True), 
            fake.words(1,religion,True), 
            fake.sentence(ext_word_list=word_list),
            fake.phone_number(),
            fake.email()] for _ in range(n)],
            columns=['Name', 'Age_profile','Age_inference', 'Gender_profile','Gender_inference','Address_actual',
                 'Address_infer','Job_profile','Job_infer',
                 'ethinictity_profile','ethinicity_infer','status_profile','status_infer', 
                 'interest_profile','interest_infer','Orientation_profile','Orientation_infer',
                 'Drinking_profile','Smoking_profile','Religion_profile','Religion_infer',
                 'Description','Phone number', 'Email ID'])
