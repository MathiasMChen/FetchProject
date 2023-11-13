# FetchProject

This project is expected to return output estimating the monthly scanned receipts. It allows users to input their desired month in given format to get a prediction for that month.

Choice of model: I ran some simple models in excel on the provided csv file and see that it is close to linear.

ML technique: I used an algorithm that starts from random parameters and move a bit closer to provided data every loop. At thousands of repeats, it can finally reach a small bias/error so that the yielded prediction model closely fits the actual data as much as possible.

How to run:

1. Download the whole directory.
2. Make sure you have Python in your computer. If not, you can go to https://www.python.org/downloads/ for download.
3. Open Windows Powershell.
4. Type "python WebBase.py". You should see messages signaling that it is running.
5. After about 10 seconds, you are expected to see the message "Running on http://127.0.0.1:5000/".
6. Go to http://127.0.0.1:5000/. You can see a figure with an interactive input box that allows you to get prediction of any month within the range.
