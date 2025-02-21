from clean.main import JsonClean as clean

from single_feedback.main import AIFeedback

# # clean data
# for i in range(10):
#     instance = clean(i)
#     instance.main()

main = AIFeedback()
main.main()