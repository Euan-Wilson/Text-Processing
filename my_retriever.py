import math

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.idfs = self.inverse_document_frequency()
        self.doc_size = self.document_vector_size()
        
        
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    
    
    # Precalculates the idf values for all of the terms
    def inverse_document_frequency(self):
      #  doc_freq = number of documents containing the word
      
      idfs = {}
      for term in self.index:
          
          doc_freq = len(self.index[term].keys())
          inv_doc_freq = math.log10(self.num_docs / doc_freq)
          idfs[term] = inv_doc_freq
          
          
      return idfs

        
    
    # Function to calculate the document vector size 
    # For the three different term weighting schemes
    def document_vector_size(self):

        doc_size = {}
        if self.term_weighting == 'binary':
            for term, docs in self.index.items():
                for doc_id in docs.keys():
                    # print(docs.values())
                    if doc_id not in doc_size:
                        doc_size[doc_id] = 0
                    doc_size[doc_id] += 1
                    
            

        elif self.term_weighting == 'tf':
            for term, docs in self.index.items():
                    for doc_id, counts in docs.items():
                        if doc_id not in doc_size:
                            doc_size[doc_id] = 0
                        doc_size[doc_id] += counts ** 2

        elif self.term_weighting == 'tfidf':
            for term, docs in self.index.items():
                    for doc_id, counts in docs.items():
                        if doc_id not in doc_size:
                            doc_size[doc_id] = 0
                        doc_size[doc_id] += self.idfs[term] * counts
            
                
        for doc_id, counts in doc_size.items():
                doc_size[doc_id] = math.sqrt(counts)
        
          
        
        # print(doc_size)
        return(doc_size)
        
        

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).

    def for_query(self, query):    
        queries = {}
        documents = self.index

        # Converting each query to a dictionary
        # With the terms as keys and frequency as values
        for term in query:
            word = term
            if word not in queries:
                queries[word] = 0
            queries[word] += 1
            
            
           
        # alldocs are the relevant doc ids for each query
        # so now need to only use these doc ids for similarity
        alldocs = {}
        for term in documents:
            if term in queries.keys():
                alldocs = alldocs | documents[term].keys()
        # print(alldocs)
                # print(term,"=",documents[term].keys())

        
        sum_qd = 0    
        
        if self.term_weighting == 'binary':
            
            # Calculates the sum of the query * document for each term
            # Using a binary term weighting i.e. either 1 for present
            # or zero for not present
            sum_qd = {}
            for docid in alldocs:
                sum_qd[docid] = 0
                for term, counts in queries.items():
                    if term in self.index:
                        if docid in self.index[term].keys():
                            sum_qd[docid] += 1
                        
                        # print(sum_qd)

        if self.term_weighting == 'tf':
            sum_qd = {}
            for docid in alldocs:
                sum_qd[docid] = 0
                for term, counts in queries.items():
                    if term in self.index:
                        if docid in self.index[term].keys():
                           
                            # Calculating the sum of the 
                            # query frequency * document frequency for each term 
                            # in the query
                            sum_qd[docid] += self.index[term][docid] * counts

            
# Calculate the tfidf values for the query & document terms
# then use these to calcualte the sum of query tfidf * document tfidf
        if self.term_weighting == 'tfidf':
            

            query_tfidf = {}
            doc_tfidf = {}

            sum_qd = {}
            for docid in alldocs:
                sum_qd[docid] = 0
                for term, counts in queries.items():
                    query_tfidf[term] = 0
                    if term in self.index:
                        # print(term, self.idfs[term])
                        query_tfidf[term] += (counts * self.idfs[term])
                        # print(term, query_tfidf[term])
                        if docid in self.index[term].keys():
                                # print(doc_tfidf[term][docid])
                                doc_tfidf[docid] = self.index[term][docid] * self.idfs[term]
                                sum_qd[docid] += doc_tfidf[docid] * query_tfidf[term]
        
            
           
        # print(doc_tfidf)
        cosine = {}
        doc_size = self.document_vector_size()
        for key in sum_qd.keys():
            # print(key)
            cosine[key] = 0
            if key in doc_size.keys():
                cosine[key] = format((sum_qd.get(key) / doc_size.get(key)), '.3f')
        
            
        import operator
        
        sorted_d = dict(sorted(cosine.items(), key=operator.itemgetter(1),reverse=True))
        # print(sorted_d)
        top_docids = list(sorted_d.keys())
        # print(top_docids)
        top_ten = top_docids[:10]
        # print(top_ten)
                
            
        return top_ten