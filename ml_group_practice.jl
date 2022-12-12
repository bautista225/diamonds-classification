# Loading assignments functions.

include("assignments_functions.jl");
include("practice_functions.jl");

# Importing modules

#import Pkg; 
#Pkg.add("CSV");
#Pkg.add("DataFrames");

using CSV;
using DataFrames;

# Loading csv file as dataframe.

df = DataFrame(CSV.File("diamonds.csv"))

describe(df)

# As Column1 has the same indices as dataframe indices, we delete Column1.
df_filtered = df[!,names(df)[names(df) .!= "Column1"]]

# Rename of some columns in order to give more information about what contains each feature.
rename!(df_filtered, Dict(:depth => :total_depth_percentage, :x => :length, :y => :width, :z => :depth))

# Getting pointers of categorical columns and numerical columns.

categorical_columns, numerical_columns = getCategoricalNumericalFeatures(df_filtered)

println("Categorical features: $(categorical_columns)")
println("Numerical features: $(numerical_columns)")

# Checking if there are numerical features with ceros when they shouldn't. 

zerosPerColumn = getNumberOfZerosPerColumn(df_filtered, numerical_columns)

for column in keys(zerosPerColumn)    
    numberOfZeros, lengthOfColumn = zerosPerColumn[column]
    println("Zeros for column $(column): $(numberOfZeros) over a total of $(lengthOfColumn)")
end

# Deleting rows with zeros, as they are an small number over the size of the dataset .

df_filtered_without_zeros = deleteRowsWithCeros(df_filtered, numerical_columns)

describe(df_filtered_without_zeros)

for column in numerical_columns

    column_data = df_filtered_without_zeros[!,column]

    (outliers_indices, extreme_outliers_indices) = getBoundariesIndices(column_data)

    println("Outliers for column $(column): $(length(column_data[outliers_indices])) over $(length(column_data))")
    println("Extreme-outliers for column $(column): $(length(column_data[extreme_outliers_indices])) over $(length(column_data))\n")
end

numberOfOutliers, numberOfExtremeOutliers = getNumberOfOutliers(df_filtered_without_zeros, numerical_columns)

println("Rows to delete for extreme outliers $(numberOfExtremeOutliers)")
println("Rows to delete for extreme outliers $(numberOfOutliers)")

df_filtered_without_outliers = deleteOutliers(df_filtered_without_zeros, numerical_columns, deleteOnlyExtremeOutliers=false)

println("Rows left $(size(df_filtered_without_outliers, 1))")

# Delete Duplicates

numberOfDuplicates = getNumberOfDuplicates(df_filtered_without_outliers)

println("Number of duplicated entries: $(numberOfDuplicates)")

df_without_duplicates = deleteDuplicatedRows(df_filtered_without_outliers)

columns_to_previsualize = [:cut, :total_depth_percentage_check, :total_depth_percentage, :length, :depth, :width]

error_rate = 0.05

error_indices, df_with_check = getTotalDepthPercentageErrorIndices(df_without_duplicates, error_rate=error_rate)

println("Rows with a relative error greater or equal to $(error_rate*100)% :")
display(df_with_check[error_indices, columns_to_previsualize])

df_without_relative_errors = deleteTotalDepthPercentageErrors(df_without_duplicates, error_indices)

println("Dataset without relative errors $(size(df_without_relative_errors,1)) over $(size(df_without_duplicates,1)).")

# Volume variable

df_with_volume = addVolumeColumntoDataframe(df_without_relative_errors)
push!(numerical_columns, :volume)

# plot_dataframe(df_with_volume, categorical_columns, numerical_columns, save_charts=true)

df_preprocessed = copy(df_with_volume)

# Ordinal enconding

df_ordinal_encoded = diamondsOrdinalEncoding(df_preprocessed)

# OneHot encoding for target variable

targets = df_preprocessed[!,:cut]
targets = String15.(targets)

input_data = df_ordinal_encoded[!, Not("cut")]
output_data = targets

Random.seed!(123)

testRatio = 0.1
N = size(input_data,1)

(trainingIndex, testIndex) = holdOut(N, testRatio)

train_input, train_output   = (input_data[trainingIndex,:], output_data[trainingIndex,:])
test_input, test_output     = (input_data[testIndex,:],     output_data[testIndex,:])

# Preserve initial indexes.

column_names = names(train_input)

numerical_indices = indexin(["carat", "total_depth_percentage", "table", "price", "length", "width", "depth", "volume"], column_names)
ordinal_indices = indexin(["color","clarity"], column_names)

# Normalize numeric variables (zero mean)

normalization_parameters = getZeroMeanNormalizationParameters(train_input, numerical_columns) # Only with train set.

train_input_numerical_columns_zeroMean  = getZeroMeanNormalizedColumns(train_input, numerical_columns, normalization_parameters)
test_input_numerical_columns_zeroMean   = getZeroMeanNormalizedColumns(test_input,  numerical_columns, normalization_parameters)

# Normalize ordinal variables (min-max)

ordinal_columns = categorical_columns[categorical_columns .!= :cut]

normalization_parameters = getMinMaxNormalizationParameters(train_input, ordinal_columns) # Only with train set.

train_input_ordinal_columns_minMax  = getMinMaxNormalizedColumns(train_input, ordinal_columns, normalization_parameters)
test_input_ordinal_columns_minMax   = getMinMaxNormalizedColumns(test_input,  ordinal_columns, normalization_parameters)

train_input_transformed = Matrix{Float64}(undef, size(train_input))
test_input_transformed  = Matrix{Float64}(undef, size(test_input))

train_input_transformed[:, numerical_indices]   = train_input_numerical_columns_zeroMean
train_input_transformed[:, ordinal_indices]     = train_input_ordinal_columns_minMax

test_input_transformed[:, numerical_indices]   = test_input_numerical_columns_zeroMean
test_input_transformed[:, ordinal_indices]     = test_input_ordinal_columns_minMax

# Train
train_inputs = train_input_transformed
train_targets = train_output[:,1]

kfoldindex = crossvalidation(train_targets, 5)

# Test
test_inputs = test_input_transformed
test_targets = test_output[:, 1]

# SVM classifier

# 8 different models varing C and kernel
parameters = [
    Dict("kernel" => "rbf",     "C" => 0.1,),
    Dict("kernel" => "poly",    "C" => 0.1,),
    Dict("kernel" => "sigmoid", "C" => 0.1,),
    Dict("kernel" => "rbf",     "C" => 1,),
    Dict("kernel" => "poly",    "C" => 1,),
    Dict("kernel" => "sigmoid", "C" => 1,),
    Dict("kernel" => "rbf",     "C" => 10,),
    Dict("kernel" => "poly",    "C" => 10,),
    Dict("kernel" => "sigmoid", "C" => 10,),
];

common_parameters = Dict("degree" => 3, "gamma" => "scale")

println("Cross-validating SVM classifier...")

models_performance = performCrossValidationTests(parameters, common_parameters, :SVM, train_inputs, train_targets, kfoldindex)

svm_best_model_parameters = getBestModelParameters("Fscore", models_performance, parameters, common_parameters)

# DecisionTree classifier

# 6 different models varing depth
parameters = [
    Dict("max_depth" => 5,),
    Dict("max_depth" => 7,),
    Dict("max_depth" => 10,),
    Dict("max_depth" => 12,),
    Dict("max_depth" => 14,),
    Dict("max_depth" => 16,),
];

common_parameters = Dict("random_state" => 318)

println("Cross-validating decision tree classifier...")

models_performance = performCrossValidationTests(parameters, common_parameters, :DecisionTree, train_inputs, train_targets, kfoldindex)

dt_best_model_parameters = getBestModelParameters("Fscore", models_performance, parameters, common_parameters)

# kNN classifier

# 6 differentt models varing k values
parameters = [
    Dict("k" => 3),
    Dict("k" => 5),
    Dict("k" => 7),
    Dict("k" => 9),
    Dict("k" => 11),
    Dict("k" => 13),
];

common_parameters = Dict()

println("Cross-validating kNN classifier...")

models_performance = performCrossValidationTests(parameters, common_parameters, :kNN, train_inputs, train_targets, kfoldindex)

knn_best_model_parameters = getBestModelParameters("Fscore", models_performance, parameters, common_parameters)

# ANN model

# 8 different models varing topology (1-2 hidden layers), learning rate...
parameters = [
    Dict(
        "topology"              => [8],
        "transferFunctions"     => [σ],
        "learningRate"          => 0.01
    ),
    Dict(
        "topology"              => [20],
        "transferFunctions"     => [σ],
        "learningRate"          => 0.01
    ),
    Dict(
        "topology"              => [8],
        "transferFunctions"     => [σ],
        "learningRate"          => 0.05
    ),
    Dict(
        "topology"              => [20],
        "transferFunctions"     => [σ],
        "learningRate"          => 0.05
    ),
    Dict(
        "topology"              => [7,7],
        "transferFunctions"     => [σ, σ],
        "learningRate"          => 0.01
    ),
    Dict(
        "topology"              => [8,4],
        "transferFunctions"     => [σ, σ],
        "learningRate"          => 0.01
    ),
    Dict(
        "topology"              => [7,7],
        "transferFunctions"     => [σ, σ],
        "learningRate"          => 0.05
    ),
    Dict(
        "topology"              => [8,4],
        "transferFunctions"     => [σ, σ],
        "learningRate"          => 0.05
    ),
];

common_parameters = Dict(
    "showText" => false,
    "maxEpochs"             => 1000,
    "minLoss"               => 0.0,
    "repetitionsTraining"   => 2,
    "validationRatio"       => 0.15,
    "maxEpochsVal"          => 20,
)

println("Cross-validating ANN classifier...")

models_performance = performCrossValidationTests(parameters, common_parameters, :ANN, train_inputs, train_targets, kfoldindex)

ann_best_model_parameters = getBestModelParameters("Fscore", models_performance, parameters, common_parameters)

# Performing Weighted voting classifier ensemble.

println("Training weighed voting ensemble classifier...")

ensemble_model = trainClassEnsemble(
    [:SVM, :DecisionTree, :kNN, :MLPC], 
    [svm_best_model_parameters, dt_best_model_parameters, knn_best_model_parameters, ann_best_model_parameters],
    (train_inputs,train_targets),
    [3,1,1,2]
)

# Calculate metrics with confussionMatrix() function
(acc,_,sensitivity,specificity,PPV,NPV,f1,_) = confusionMatrix(predict(ensemble_model, test_inputs),test_targets)

pretty_table(header=["Final Model","Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Fscore"],hcat(
    "Final Ensemble",
    acc,sensitivity,specificity,PPV,NPV,f1
    )  
)

cut_range_ordered = ["Fair", "Good", "Very Good", "Premium", "Ideal"]

train_targets_onehot = oneHotEncoding(train_targets, cut_range_ordered)
test_targets_onehot = oneHotEncoding(test_targets, cut_range_ordered)

println("Training best ANN...")

(ann,_) = trainClassANN(ann_best_model_parameters["topology"],  
    (train_inputs, train_targets_onehot),
    testDataset = (test_inputs, test_targets_onehot), 
    transferFunctions = ann_best_model_parameters["transferFunctions"], 
    maxEpochs = ann_best_model_parameters["maxEpochs"], minLoss = ann_best_model_parameters["minLoss"], 
    learningRate = ann_best_model_parameters["learningRate"],
    maxEpochsVal=20, showText=false)

# Calculate metrics with confussionMatrix() function
(acc,_,sensitivity,specificity,PPV,NPV,f1,_) = confusionMatrix(ann(test_inputs')',test_targets_onehot)

pretty_table(header=["Final Model","Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Fscore"],hcat(
    "Final ANN",
    acc,sensitivity,specificity,PPV,NPV,f1
    )  
)

# ScikitLearn models (SVM)
println("Training best SVM...")
model = trainClassSklearn(:SVM, svm_best_model_parameters, train_inputs, train_targets)

# Calculate metrics with confussionMatrix() function
(acc,_,sensitivity,specificity,PPV,NPV,f1,_) = confusionMatrix(predict(model, test_inputs),test_targets)

pretty_table(header=["Final Model","Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Fscore"],hcat(
    "Final SVM",
    acc,sensitivity,specificity,PPV,NPV,f1
    )  
)

# ScikitLearn models (Decision tree)
println("Training best decision tree...")
model = trainClassSklearn(:DecisionTree, dt_best_model_parameters, train_inputs, train_targets)

# Calculate metrics with confusionMatrix() function
(acc,_,sensitivity,specificity,PPV,NPV,f1,_) = confusionMatrix(predict(model, test_inputs),test_targets)

@sk_import metrics: confusion_matrix

display(confusion_matrix(test_targets, predict(model, test_inputs)))

pretty_table(header=["Final Model","Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Fscore"],hcat(
    "Final DT",
    acc,sensitivity,specificity,PPV,NPV,f1
    )  
)

# ScikitLearn models (KNN)
println("Training best KNN...")
model = trainClassSklearn(:kNN, knn_best_model_parameters, train_inputs, train_targets)

# Calculate metrics with confussionMatrix() function
(acc,_,sensitivity,specificity,PPV,NPV,f1,_) = confusionMatrix(predict(model, test_inputs),test_targets)

pretty_table(header=["Final Model","Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Fscore"],hcat(
    "Final kNN",
    acc,sensitivity,specificity,PPV,NPV,f1
    )  
)
