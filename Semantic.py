#!/usr/bin/env python
# coding: utf-8

# In[3]:


import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))


# In[8]:


nlp = spacy.load('en_core_web_md')

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# In[5]:


sentence_to_compare ="Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence +" - ",similarity)


# ## Similarities between Cat, Monkey and Banana
# 
# When comparing the similarities between "cat," "monkey," and "banana," one interesting
# observation is that these words are semantically related in a way that implies a common
# context: animals and food. The relationship between these words can be categorized as
# follows:
# 
# Cat and monkey: Both are animals, specifically mammals. They share similar features
# such as fur, four legs, and a tail. However, they belong to different species and have
# distinct characteristics.
# 
# Monkey and banana: This relationship is often depicted in popular culture, where
# monkeys are commonly associated with eating bananas. Monkeys are known to enjoy
# eating bananas in various media, which has led to the perception of a close connection
# between the two.
# 
# Cat and banana: While there is no inherent connection between cats and bananas in
# nature, the similarity might be derived from the fact that cats are often depicted as
# playful and curious, and they might show interest in objects like bananas due to their
# shape and movement.

# ## Making Up Example Of My Own
# 
# Let's take the words "car," "driver," and "road." In this example, we can find interesting
# similarities between these terms:
# 
# Car and driver: These two words are interconnected as a car requires a driver to operate
# it. A car without a driver is unable to function effectively. This relationship highlights 
# the essential role of a driver in controlling and maneuvering a car.
# 
# Driver and road: The connection between a driver and a road signifies their interaction in
# the context of transportation. A driver relies on roads to navigate and reach their desired
# destinations. The condition and quality of the road impact the driver's experience and the
# overall safety of the journey.
# 
# Car and road: The relationship between a car and a road lies in their mutual dependency.
# A car relies on the road to provide a surface for travel, while the road is designed to
# accommodate cars and other vehicles. They form a fundamental infrastructure for
# transportation.

# In[11]:


# Running the example file with 'en_core_web_sm'

nlp = spacy.load('en_core_web_sm')

# Below is a list of six complaints.
complaints = [ 'We bought a house in  CA. Our mortgage was handled by a company called ki. Soon after the mortgage was sold to ABC. Shortly after that XYZ took over the mortgage. The other day we got a notice not to send our payment to them but to loi instead. This is all so frustrating and wreaks of the  mortgage nightmare.',
'I got approved for a loan to buy a house I have submitted everything I need to for them I paid for the inspection and paid good faith check after all of that they said I did not get approved for the loan to cancel my contract because they do not want to wait for the down payments assistant said that the Sellers do not want to wait that long I feel like they are getting over on me I feel that they should have told me that I did not get approved before I spent my money and picked out a house Carrington mortgage in Ohio ',
'As per the correspondence, I received from : The University  This is to inform you that I have recently pulled my credit report and noticed that there is a collection listing from The University  on my credit report. I WAS never notified of this collection action or that I owed the debt. This letter is to inform you that I would like a verification of the debt and juilo ability to collect this money from me.',
'I am writing to dispute the follow information in my file.ON BOTH TransUnion & . for {$15000.00}. I have contacted this agency to advise to STOP CALLING ME this case was dismissed in court  2014. Please see the attached document from  County State Court. Thanking you in advanced regarding this matter.',
'I have not had a XXXX phone since early 2007. I have tried to resolve my bill in the past but it keeps reposting an old bill. I have no way to provide financial info from 8 years ago and they know that so they want me to prove it to them but I have no way to do that. Is there anyway to get  to find out how old it is.',
'I posted dated a check and mailed it for 2015 for my mortgage payment as my mortgage company will only take online payments if all the late charges are paid at once ( also illegal ), and the check was cashed on 2015 which cost me over {$70.00} in over draft fees with my bank.'
]

# We will now compare the similarity of the complaints to ascertain if spaCy's similarity
# model is able to distinguish between these long pieces of text.

print("-------------Complaints similarity---------------")
for token in complaints:
    token = nlp(token)
    for token_ in complaints:
        token_ = nlp(token_)
        print(token.similarity(token_))

# Below is a list of six recipe instructions.

recipes= [ 'Bake in the preheated oven, stirring every 20 minutes, until sugar mixture has baked and caramelized onto popcorn and cashews, about 1 hour. Spread cashew caramel corn onto a parchment paper-lined baking sheet to cool. If desired, form into balls while still warm.',
'Combine brown sugar, corn syrup, butter, salt, and cream of tartar in a large saucepan. Bring to a boil, stirring constantly, until a candy thermometer inserted into the middle of the syrup, not touching the bottom, reads 260 degrees F (127 degrees C), 6 to 8 minutes.',
'Lift marshmallow fudge out of the pan by the edges of the foil and place on a large cutting board. Dip a large knife in the remaining confectioners\' sugar and slice fudge into 1 1/2-inch squares, continually dipping knife in the sugar after each slice.',
'Melt butter in a medium saucepan over medium heat; stir in condensed milk. Pour in chocolate chips; cook and stir until melted, 5 to 10 minutes.',
'Lightly grease a cookie sheet. Deflate the dough and turn it out onto a lightly floured surface. Roll the marzipan into a rope and place it in the center of the dough. Fold the dough over to cover it; pinch the seams together to seal. Place the loaf, seam side down, on the prepared baking sheet. Cover with a damp cloth and let rise until doubled in volume, about 40 minutes. Meanwhile, preheat oven to 350 degrees F (175 degrees C)',
'In a large bowl, cream together the butter, brown sugar, and white sugar. Beat in the instant pudding mix until blended. Stir in the eggs and vanilla. Blend in the flour mixture. Finally, stir in the chocolate chips and nuts. Drop cookies by rounded spoonfuls onto ungreased cookie sheets.'
]

# We will now compare the similarity of the recipes. to ascertain how well spaCy's similarity
# model is able to distinguish between them.

print("-------------Recipes similarity---------------")
for token in recipes:
    token = nlp(token)
    for token_ in recipes:
        token_ = nlp(token_)
        print(token.similarity(token_))

# Now we want to obtain the extent of similarity between the complaints and the recipes.
# we will loop through every recipe instruction and compare it with a complaint.

print("-------------Recipes similarity---------------")

for token in recipes:
    token = nlp(token)
    for token_ in complaints:
        token_ = nlp(token_)
        print(token.similarity(token_))

# What do you observe? Note that the similarity index has reduced from what we observed in the short-text example discussed in the content PDF.


# There are several ways to make your model more accurate with the similarity
# or even prediction such as feeding it with some training data. This could include
# more vocabulary about food and recipes if you are building a models concerning food.
# You can also head over to spaCy documentation here: https://spacy.io/usage/linguistic-features#vectors-similarity
# and check out other cool stuff!


# ## Difference between 'en_core_web_sm' and 'en_core_web_md' 
# 
# The 'en_core_web_sm' model is a smaller and simpler version of the English language model
# provided by spaCy. It has a smaller file size, which makes it quicker to download and requires
# less memory to load. However, this smaller size comes with a trade-off in terms of linguistic 
# coverage and accuracy compared to the larger 'en_core_web_md' model.
# 
# When running the example file with the 'en_core_web_sm' model,I noticed a
# few differences compared to the 'en_core_web_md' model:
# 
# 1. Linguistic capabilities: The 'en_core_web_sm' model provides essential linguistic
# annotations such as tokenization, part-of-speech tagging, and dependency parsing. 
# However, it may have limited coverage for certain language phenomena and may not 
# perform as well on more complex or specialized tasks.
# 
# 2. Vocabulary and word vectors: The 'en_core_web_sm' model has a smaller vocabulary
# and word vector size compared to the 'en_core_web_md' model. This means that it 
# may not recognize or handle out-of-vocabulary words as effectively and may have 
# reduced performance on tasks that rely heavily on word embeddings or semantic 
# similarity.
# 
# 3. Accuracy and precision: Due to its smaller size, the 'en_core_web_sm' model might
# exhibit slightly lower accuracy and precision compared to the 'en_core_web_md' 
# model. It may have more difficulty in accurately disambiguating certain word senses
# or capturing nuanced linguistic phenomena.
# 
# Overall, the 'en_core_web_sm' model is designed to be a lightweight and efficient
# option for general-purpose language processing tasks, where speed and memory 
# constraints are a priority. However, if you require more comprehensive linguistic 
# analysis or plan to tackle more complex language tasks, the 'en_core_web_md' model 
# might be a better choice, despite its larger size and increased computational requirements.

# In[ ]:




