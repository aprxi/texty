#!/usr/bin/env python3
"""Dutch stopwords list for filtering."""

DUTCH_STOPWORDS = {
    # Articles
    'de', 'het', 'een',
    
    # Pronouns
    'ik', 'je', 'jij', 'u', 'hij', 'ze', 'zij', 'we', 'wij', 'jullie', 'hen', 'hun',
    'me', 'mij', 'jou', 'uw', 'hem', 'haar', 'ons', 'onze',
    'dit', 'dat', 'deze', 'die', 'welk', 'welke',
    
    # Conjunctions
    'en', 'of', 'maar', 'want', 'omdat', 'als', 'dan', 'dus', 'noch', 'zodat',
    
    # Prepositions
    'in', 'op', 'aan', 'bij', 'van', 'voor', 'naar', 'met', 'om', 'te', 'tot',
    'uit', 'over', 'door', 'na', 'sinds', 'tussen', 'onder', 'tegen', 'zonder',
    
    # Common verbs
    'is', 'zijn', 'was', 'waren', 'ben', 'bent', 'heb', 'hebt', 'heeft', 'hebben',
    'had', 'hadden', 'zal', 'zult', 'zullen', 'zou', 'zouden', 'kan', 'kunt',
    'kunnen', 'kon', 'konden', 'mag', 'mogen', 'moet', 'moeten', 'moest', 'moesten',
    'wil', 'wilt', 'willen', 'wilde', 'wilden', 'word', 'wordt', 'worden', 'werd',
    'werden', 'ga', 'gaat', 'gaan', 'ging', 'gingen', 'kom', 'komt', 'komen',
    'kwam', 'kwamen', 'doe', 'doet', 'doen', 'deed', 'deden',
    
    # Other common words
    'er', 'hier', 'daar', 'nu', 'dan', 'nog', 'al', 'wel', 'niet', 'geen',
    'ook', 'zo', 'zeer', 'meer', 'veel', 'alle', 'alles', 'niets', 'iets',
    'wat', 'wie', 'waar', 'wanneer', 'waarom', 'hoe', 'ja', 'nee',
    
    # Additional filler words
    'echter', 'toch', 'dus', 'namelijk', 'immers', 'tenslotte', 'overigens',
    'verder', 'bovendien', 'daarom', 'daardoor', 'daarmee', 'daarnaast',
    'bijvoorbeeld', 'misschien', 'ongeveer', 'eigenlijk', 'gewoon',
    
    # Numbers and time
    'eerste', 'tweede', 'derde', 'laatste', 'enkele', 'veel', 'weinig',
    'vandaag', 'gisteren', 'morgen', 'nu', 'toen', 'straks', 'altijd', 'nooit'
}

def get_dutch_stopwords():
    """Return the set of Dutch stopwords."""
    return DUTCH_STOPWORDS

def filter_dutch_stopwords(words):
    """Filter out Dutch stopwords from a list of words."""
    return [w for w in words if w.lower() not in DUTCH_STOPWORDS]