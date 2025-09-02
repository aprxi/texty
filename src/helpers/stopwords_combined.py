#!/usr/bin/env python3
"""Combined Dutch and English stopwords for the classifier"""

# Dutch stopwords
DUTCH_STOPWORDS = {
    # Articles & determiners
    'de', 'het', 'een', 'der', 'den', 'des', 
    
    # Personal pronouns
    'ik', 'je', 'jij', 'u', 'hij', 'ze', 'zij', 'we', 'wij', 'jullie', 'hen', 'hun',
    'me', 'mij', 'hem', 'haar', 'ons', 'jou', 'jouw', 'uw',
    
    # Demonstrative pronouns
    'dit', 'dat', 'deze', 'die', 'daar', 'hier',
    
    # Common verbs
    'is', 'was', 'zijn', 'waren', 'ben', 'bent', 'heb', 'hebt', 'heeft', 'hebben', 
    'had', 'hadden', 'word', 'wordt', 'worden', 'werd', 'werden', 'zal', 'zult', 
    'zullen', 'zou', 'zouden', 'kan', 'kunt', 'kunnen', 'kon', 'konden', 'mag', 
    'mogen', 'moet', 'moeten', 'moest', 'moesten', 'doe', 'doet', 'doen', 'deed', 
    'deden', 'ga', 'gaat', 'gaan', 'ging', 'gingen',
    
    # Prepositions
    'in', 'op', 'aan', 'bij', 'van', 'voor', 'met', 'om', 'door', 'over', 'uit', 
    'naar', 'te', 'ter', 'tot', 'tegen', 'onder', 'tussen', 'binnen', 'zonder',
    'tijdens', 'na', 'naast', 'langs', 'rond', 'boven', 'beneden', 'achter',
    
    # Conjunctions
    'en', 'of', 'maar', 'want', 'omdat', 'als', 'dan', 'dus', 'noch', 'echter',
    'toch', 'hoewel', 'ofschoon', 'daarom', 'daardoor',
    
    # Common adverbs
    'niet', 'wel', 'zeer', 'veel', 'meer', 'meest', 'al', 'reeds', 'nog', 'steeds',
    'weer', 'er', 'daar', 'hier', 'waar', 'nu', 'toen', 'altijd', 'nooit', 'vaak',
    'soms', 'misschien', 'graag', 'even', 'ongeveer', 'vooral', 'bijvoorbeeld',
    
    # Question words
    'wat', 'wie', 'waar', 'wanneer', 'waarom', 'hoe', 'welke', 'welk',
    
    # Other common words
    'geen', 'iets', 'iemand', 'niets', 'niemand', 'alles', 'iedereen', 'andere',
    'zelf', 'eigen', 'zo', 'ook', 'alleen', 'alle', 'elke', 'elk', 'enkele',
}

# English stopwords
ENGLISH_STOPWORDS = {
    # Articles & determiners
    'a', 'an', 'the', 'this', 'that', 'these', 'those',
    
    # Personal pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
    
    # Common verbs
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'shall', 'can', 'cannot', "can't", "won't", "wouldn't", "couldn't", "shouldn't",
    'get', 'gets', 'got', 'getting', 'go', 'goes', 'went', 'going', 'gone',
    
    # Prepositions
    'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about', 'into',
    'through', 'over', 'under', 'between', 'among', 'after', 'before', 'during',
    'without', 'within', 'along', 'against', 'around', 'behind', 'beneath', 'beside',
    
    # Conjunctions
    'and', 'or', 'but', 'if', 'because', 'as', 'while', 'when', 'where', 'so',
    'though', 'although', 'whether', 'unless', 'since', 'until', 'than', 'that',
    
    # Common adverbs
    'not', 'no', 'yes', 'very', 'just', 'only', 'also', 'too', 'so', 'very',
    'now', 'then', 'here', 'there', 'where', 'when', 'why', 'how', 'all', 'both',
    'each', 'more', 'most', 'less', 'least', 'much', 'many', 'few', 'some', 'any',
    
    # Other common words
    'what', 'which', 'who', 'whom', 'whose', 'why', 'how', 'when', 'where',
    'other', 'another', 'such', 'same', 'own', 'self', 'every', 'each', 'either',
    'neither', 'both', 'all', 'any', 'some', 'none', 'one', 'two', 'three',
}

# Combined stopword set
COMBINED_STOPWORDS = DUTCH_STOPWORDS | ENGLISH_STOPWORDS

# Also add single letters that might appear
SINGLE_LETTERS = {chr(i) for i in range(ord('a'), ord('z')+1)}
COMBINED_STOPWORDS |= SINGLE_LETTERS

# IMPORTANT: Remove stopwords that might be meaningful in AI Act context
# These prepositions often form part of important phrases
PRESERVE_WORDS = {
    # Dutch prepositions that indicate relationships
    'voor',  # "AI voor werknemers" (AI for employees)
    'van',   # "beoordeling van studenten" (assessment of students)  
    'met',   # "identificatie met biometrie" (identification with biometrics)
    'door',  # "beslissing door AI" (decision by AI)
    'bij',   # "toepassing bij kinderen" (application with children)
    # English prepositions that matter
    'for',   # "AI for hiring"
    'with',  # "surveillance with facial recognition"
    'by',    # "assessment by algorithm"
    'of',    # "analysis of behavior"
}

# Remove preserved words from stopwords
COMBINED_STOPWORDS -= PRESERVE_WORDS

# Convert to list for sklearn
STOPWORDS_LIST = sorted(list(COMBINED_STOPWORDS))

if __name__ == "__main__":
    print(f"Total stopwords: {len(COMBINED_STOPWORDS)}")
    print(f"Dutch stopwords: {len(DUTCH_STOPWORDS)}")
    print(f"English stopwords: {len(ENGLISH_STOPWORDS)}")
    print("\nSample Dutch stopwords:", sorted(list(DUTCH_STOPWORDS))[:20])
    print("\nSample English stopwords:", sorted(list(ENGLISH_STOPWORDS))[:20])