# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:03:41 2018

@author: Erman
"""

import nltk
import numpy as np
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class BellaRev():
    
    
    def __init__(self):
        self.df = pd.read_excel('Semantria_Belagio_Sample_Dataset.xlsx', sheetname='sentiment_bellagio')
        self.positives = []
        self.negatives = []
        
        
    def dfToList(self,dfToBeConverted):
        resultList=list(dfToBeConverted)
        resultList+=(list(dfToBeConverted[resultList[0]]))
        return resultList

    def positivesNegatives(self):
        self.positives=self.dfToList(pd.read_csv("words-positive.csv"))
        self.negatives=self.dfToList(pd.read_csv("words-negative.csv"))
        
    def avgSentiment(self,text):
        temp=[]
        text_sent=nltk.sent_tokenize(text)
        for sentence in text_sent:
            n_count=0
            p_count=0
            
            sent_words=nltk.word_tokenize(sentence)
            
            for word in sent_words:
                if word.lower() in self.positives:
                        p_count+=1
                if word.lower() in self.negatives:
                        n_count+=1
                        
            if (p_count>0 and n_count==0):
                temp.append(1)
            elif n_count%2>0:
                temp.append(-1)
            
            elif (n_count%2==0 and n_count>0):
                temp.append(1)
            else:
                temp.append(0)
        return round(np.average(temp),2)
    
    def tf(self,review):
        tokens=nltk.word_tokenize(review)
        tokens=list(map (lambda x: x.lower(), tokens ))
        tokens=list(filter (lambda x: x.isalpha(), tokens ))
        fd=nltk.FreqDist(tokens)
        return fd
    
    def idf(self,reviewS,term):
        count=[]
        
        for review in reviewS:
            
            tokens=nltk.word_tokenize(review)
            tokens=list(map (lambda x: x.lower(), tokens ))
            review=" ".join(tokens)
    #        print(review)
    #        print("####################################")
            if term in review:
                count.append(1)
            else:
                count.append(0)
        inv_F=0
        if sum(count)>0:
            inv_F=math.log(len(count)/sum(count))
                
        return round(inv_F,2)
    
    def tfidf(self,reviewS,review,n):
        term_scores={}
        review_fd=self.tf(review)
        
        for term in review_fd:
            if term.isalpha():
                idf_val=self.idf(reviewS,term)
                tf_val=review_fd[term]
                tfidf_val=idf_val*tf_val
                term_scores[term]=round(tfidf_val,2)
                
        return sorted(term_scores.items(),key=lambda x:-x[1])[:n]
    
    
    def getWordsAndScores(self,tfidfScores):
    
        words=[tupl[0] for tupl in tfidfScores]
        scores=[tupl[1] for tupl in tfidfScores]
        
        return words,scores
    
    def autolabel(self,rects,labels,ax):
        """
        Attach a text label above each bar displaying its height
        """
        for ind in range(len(rects)):
            rect=rects[ind]
            label=labels[ind]
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.03*height,
                     label,
                    ha='center', va='bottom',size='smaller')


    def decoratedBars(self,values,labels,legendHere,myTitle,myYLabel):
            
        ind = np.arange(len(values))  # the x locations for the groups
        width = 0.35       # the width of the bars
        
        fig, ax = plt.subplots()
        rects = ax.bar(ind, values, width, color='r')
            
        # add some text for labels, title and axes ticks
        ax.set_ylabel(myYLabel)
        ax.set_title(myTitle)
        ax.set_xticks(ind )
        
        ax.legend([legendHere])
        self.autolabel(rects,labels,ax)
        plt.show()
        
    def getSentiments(self):
        self.df['Sentiments']=self.df['Text'].apply(lambda x: self.avgSentiment(x))
        self.df=self.df.sort_values(by=['Sentiments'])

        
    def printTop(self,n):
        sentiments=list(self.df['Sentiments'])
        texts=list(self.df['Text'])
        for ind in range(n):
            print(texts[ind])
            print(sentiments[ind])
        
        for ind in range(n):
            print(texts[-ind])
            print(sentiments[-ind])
            
        
            
    def getSentimentsBySource(self):
        meanSentiments=self.df.groupby(['Booking Source'])['Sentiments'].mean()
        meanSentiments=meanSentiments.sort_values(ascending=False)
        sources=list(meanSentiments.index.values)
        sents=list(meanSentiments)
        self.decoratedBars(sents,sources,"Average Sentiments","Bellagio Reviews","Sentiment Values")
        
    def getSentimentsByCity(self):
        meanSentiments=self.df.groupby(['City'])['Sentiments'].mean()
        meanSentiments=meanSentiments.sort_values(ascending=False)
        sources=list(meanSentiments.index.values)
        sents=list(meanSentiments)
        self.decoratedBars(sents,sources,"Average Sentiments","Bellagio Reviews","Sentiment Values") 
        
    def getSentimentsByManagers(self):
        meanSentiments=self.df.groupby(['Manager on Duty'])['Sentiments'].mean()
        meanSentiments=meanSentiments.sort_values(ascending=False)
        sources=list(meanSentiments.index.values)
        sents=list(meanSentiments)
        self.decoratedBars(sents,sources,"Average Sentiments","Bellagio Reviews","Sentiment Values") 
        
    def plotImportantTerms(self,posNeg,reviewNumber,wordNumber):
        
        reviewSentDict={}
        if posNeg==1:
            
            texts=list(self.df['Text'])[-reviewNumber:]
            
            for review in texts:
                words,scores=self.getWordsAndScores(self.tfidf(list(self.df['Text']),review,wordNumber))
                for ind in range(len(words)):
                    if words[ind] not in reviewSentDict:
                        reviewSentDict[words[ind]]=scores[ind]
                        
        else:
            texts=list(self.df['Text'])[:reviewNumber]
            
            for review in texts:
                words,scores=self.getWordsAndScores(self.tfidf(list(self.df['Text']),review,wordNumber))
                for ind in range(len(words)):
                    if words[ind] not in reviewSentDict:
                        reviewSentDict[words[ind]]=scores[ind]
                        
        finalWords,finalScores=self.getWordsAndScores(sorted(reviewSentDict.items(),key=lambda x:-x[1]))
        
        if posNeg==1:
            self.decoratedBars(finalScores,finalWords,"Positive Word Importances","Bellagio Reviews","tf-Idf")
        else:
            self.decoratedBars(finalScores,finalWords,"Negative Word Importances","Bellagio Reviews","tf-Idf")
            
                

            
        
        
    
    
if __name__ == '__main__':
    BRev=BellaRev()
    BRev.positivesNegatives()
    BRev.getSentiments()
    BRev.getSentimentsByCity()
    BRev.getSentimentsBySource()
    BRev.getSentimentsByManagers()
    BRev.printTop(10)
    BRev.plotImportantTerms(0,3,3)

            