import cohere
import os

co = cohere.Client('PsMBI2NlLAh8ZONu3j2x8UNZgEsj9lxZHgojVnn2')
response = co.generate(
  model='large',
  prompt='Simplify this passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn’t the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to “the dusty section of the dictionary” --',
  max_tokens=100,
  temperature=0.4,
  k=0,
  p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop_sequences=["--"],
  return_likelihoods='NONE')
print('Prediction: {}'.format(response.generations[0].text))