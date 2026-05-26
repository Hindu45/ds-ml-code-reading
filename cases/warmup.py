# %%
import numpy as np
messages = [
    "Happy learning, happy life!",
    "Data speaks louder than opinions.",
    "Data shows trends, individual predictions may be wrong.",
    "Every dataset tells a story.",
    "What do you call a data scientist with no data? A philosopher.",
    "Why did the random forest win the award? Best ensemble performance.",
    "How many data scientists does it take to change a light bulb? Just one, but first they need 10,000 labeled examples.",
    "I used to be a statistician, but I found it too mean.",
    "An SQL query walks into a bar. Walks right back out. Couldn't find a table.",
    "There are 10 types of people: those who understand binary, and those who don't.",
    "I told my bosses the model overfits. They said 'great, ship it!'",
    "Why did the neural network go to therapy? Too many hidden layers."
]
r = np.random.randint(len(messages))
print(messages[r])
