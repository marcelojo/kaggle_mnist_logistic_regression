clear
clc

% Load training data
images_train = dlmread('../train.csv', ',', 1,0);

% Labels are the first columns
labels_train = images_train(:,1);

% Copy data from second column
images_train = images_train(:,2:end);

% Just to be faster, we can train only in a part of the entire trainind dataset
train_qty = 10000;   % Max of 42000

data_train = images_train(1:train_qty, :);
label_train = labels_train(1:train_qty);


% Training
a_param = train_regression(data_train, label_train);

% Classifing
right = 0;
wrong = 0;
for i = train_qty + 1:size(images_train)(1)
    [prob, pred] = test_regression(a_param, images_train(i,:));
    
    if(pred == labels_train(i))
        right = right + 1;
    else 
        wrong = wrong + 1;
    end
endfor

accuracy = right / (right + wrong);
printf("Hits: %d, Miss: %d. Total: %d\n", right, wrong, right + wrong);
printf("Multi-class Logistic Regression accuracy: %1.2f%%\n\n", accuracy * 100);


% Load data
images_test = dlmread('../test.csv', ',', 1,0);

pred = zeros(size(images_test),2);

for i = 1:size(images_test)(1)
    pred(i,1) = i;
    [prob, pred(i,2)] = test_regression(a_param, images_test(i,:));
endfor

% Output file
dlmwrite("results.csv", pred, ',');
