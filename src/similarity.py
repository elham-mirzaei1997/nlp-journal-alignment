from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(a, b):
    return cosine_similarity(a, b)[0]