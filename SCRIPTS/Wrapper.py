 def rfe(len):
    # # https: // towardsdatascience.com / feature - selection - with-pandas - e3690ad8504b
    # #no of features
    # nof_list=np.arange(1,len)
    # high_score=0
    # #Variable to store the optimum features
    # nof=0
    # score_list =[]
    # for n in range(len(nof_list)):
    #     X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    #     model = LinearRegression()
    #     rfe = RFE(model,nof_list[n])
    #     X_train_rfe = rfe.fit_transform(X_train,y_train)
    #     X_test_rfe = rfe.transform(X_test)
    #     model.fit(X_train_rfe,y_train)
    #     score = model.score(X_test_rfe,y_test)
    #     score_list.append(score)
    #     if(score>high_score):
    #         high_score = score
    #         nof = nof_list[n]
    # print("Optimum number of features: %d" %nof)
    # print("Score with %d features: %f" % (nof, high_score))
    pass