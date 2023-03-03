from flask import Flask, request

from compute_score import compute_score


app = Flask(__name__)

@app.route('/site_eval', methods=['POST'])
def my_api():
    if request.method == 'POST':
        # Extract the data from the request
        data = request.get_json()
        # Call your function to calculate score for transportation and business
        transporation_result = compute_score(data, 'transportation')
        business_result = compute_score(data, 'business')
        # Return the result as a JSON response
        return {'transporation': transporation_result, 'business':business_result}

if __name__ == '__main__':
    app.run(debug=True)