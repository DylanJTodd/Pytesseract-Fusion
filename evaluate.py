import pytesseract
from PIL import Image, ExifTags
import cv2

# Computes the distance between two strings according to the number of single character edits (insertions, deletions, substitutions) required to change one string into the other
def levenshtein_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,        #Deletions
                           dp[i][j - 1] + 1,        #Insertions
                           dp[i - 1][j - 1] + cost) #Substitutions

    return dp[m][n]

# Computes the similarity between two strings as a percentage by normalizing the Levenshtein distance
def levenshtein_similarity_percentage(str1, str2):
    distance = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    similarity = (1 - distance / max_len) * 100
    return similarity