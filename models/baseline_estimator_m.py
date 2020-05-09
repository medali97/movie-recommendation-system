def baseline_estimator(user_id, movie_id, data):
    overall_rating_average = data['Rating'].mean()
    bias_rating_user = data.loc[data['Customer_id'] == user_id]["Rating"].mean() - overall_rating_average
    bias_rating_movie = data.loc[data['Movie_id'] == movie_id]["Rating"].mean() - overall_rating_average
    bui = overall_rating_average + bias_rating_user + bias_rating_movie
    if(bui<1):
        print("rating less than 1")
        bui = 1
    elif(bui>5):
        print("rating greater than 5")
        bui = 5
    return bui