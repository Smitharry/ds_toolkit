# Test client. Make TextCNN instance.
if __name__ == "__main__":
    import pickle
    from CNN import TextCNN
    import keras.backend as K
    from sklearn.metrics import classification_report

    text_cnn = TextCNN(15, 200, 5000, 13, 100)
    model = text_cnn.get_model()
    # text_cnn.save_to_png()

    with open('D:/users/mkuznetsova/PycharmProjects/ds_toolkit/dataset.pickle', 'rb') as infile:
        # dataset contains following files in written order: X_train, X_test, y_train, y_test
        dataset = pickle.load(infile)

    history = model.fit(dataset[0],
                        dataset[2],
                        epochs=5,
                        batch_size=64,
                        verbose=1)
    print(classification_report(dataset[3], K.eval(K.argmax(model.predict(dataset[1])))))
