# Misclassification Analysis for Best Model (TF-IDF + Logistic Regression)

Out of **199** test samples, **44** were misclassified.
Below are 10 illustrative misclassified examples along with detailed error rationales:

| # | Aspect | Student Comment | Actual Sentiment | Predicted Sentiment | Primary Error Reason |
|---|---|---|---|---|---|
| 1 | teaching | "good and qualified teachers but few lacks in interaction with students and fails to describe a topic" | `neutral` | `positive` | Neutral boundary ambiguity: Feedback expresses mild critique or conditional praise, blurring the neutral threshold. |
| 2 | teaching | "some of our faculty member are not so experienced that's why most of the time we are facing problems subjects like CSE(c++)" | `negative` | `neutral` | Complex negation / nuance: Negation syntax confused standard token weights or context. |
| 3 | teaching | "teaching delivery is good .but it is better to reduce there speed of  lecture delivery to understand the student better" | `neutral` | `positive` | Neutral boundary ambiguity: Feedback expresses mild critique or conditional praise, blurring the neutral threshold. |
| 4 | teaching | "I am very much satisfied with teaching faculty interms of iteraction,lecture delivery, punctuality etc" | `positive` | `neutral` | Neutral boundary ambiguity: Feedback expresses mild critique or conditional praise, blurring the neutral threshold. |
| 5 | coursecontent | "It was good but can be better" | `neutral` | `positive` | Neutral boundary ambiguity: Feedback expresses mild critique or conditional praise, blurring the neutral threshold. |
| 6 | coursecontent | "Content is also satisfactory" | `neutral` | `positive` | Neutral boundary ambiguity: Feedback expresses mild critique or conditional praise, blurring the neutral threshold. |
| 7 | coursecontent | "every thing is fine" | `positive` | `neutral` | Neutral boundary ambiguity: Feedback expresses mild critique or conditional praise, blurring the neutral threshold. |
| 8 | coursecontent | "the material provided is beneficial and provides much knowledge" | `positive` | `neutral` | Complex negation / nuance: Negation syntax confused standard token weights or context. |
| 9 | coursecontent | "not bad" | `positive` | `negative` | Extreme brevity: Text contains very few tokens, lacking explicit lexical sentiment context. |
| 10 | coursecontent | "course contents are very much relevant to our courses" | `positive` | `neutral` | Neutral boundary ambiguity: Feedback expresses mild critique or conditional praise, blurring the neutral threshold. |

## Common Failure Modes Identified:
1. **Brevity**: Single-word comments like *'Average'* or *'Fine'* can be borderline neutral vs positive depending on context.
2. **Mixed Sentiment**: Students often combine praise with criticism (*'Teacher is very polite but speed is too fast'*).
3. **Neutral Ambiguity**: Neutral class is naturally harder to separate as it borders mild negative and mild positive opinions.
4. **Sarcasm and Colloquialisms**: Sarcastic remarks (*'Great job making exams impossible'*) possess positive surface lexicon with negative intent.