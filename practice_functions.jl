##### Libraries:
# import Pkg;
# Pkg.add("PlotlyJS");

using PlotlyJS;
using Statistics;

# AdHoc functions

function ordinalEncoding(feature::Vector{<:Any}, classes::Vector{<:Any})
    mapping = Dict(classes .=> 1:length(classes))
    return map(feat -> mapping[feat], feature)
end

ordinalEncoding(feature::Vector{<:Any}) = ordinalEncoding(feature, unique(feature))

function getCategoricalNumericalFeatures(df)

    # For every column, extract a list of columns with string type.
    categorical_columns = propertynames(df)[eltype.(eachcol(df)) .<: AbstractString]

    # For every column, extract a list of columns with type different from string.
    numerical_columns = propertynames(df)[propertynames(df) .∉ Ref(categorical_columns)]

    return categorical_columns, numerical_columns
end

function getNumberOfZerosPerColumn(df, columns)

    zerosPerColumn = Dict()

    for column in columns
        column_data = df[!, column]

        numberOfZeros = length(column_data[column_data .== 0])
        lengthOfColumn = length(column_data)

        zerosPerColumn[column] = (numberOfZeros, lengthOfColumn)
    end

    return zerosPerColumn
end

function deleteRowsWithCeros(df, columns)

    df_copy = copy(df)

    for column in columns

        column_data = df[!, column]
    
        if length(column_data[column_data .== 0]) > 0
            df_copy = df[column_data .!= 0, :]
        end
    end

    return df_copy
end

function getBoundariesIndices(df_column)

    q1 = quantile(df_column, 0.25)
    q3 = quantile(df_column, 0.75)

    interquartile_range = q3 - q1
    
    upper_boundary = q3 + (1.5 * interquartile_range)
    lower_boundary = q1 - (1.5 * interquartile_range)

    extreme_upper_boundary = q3 + (3 * interquartile_range)
    extreme_lower_boundary = q1 - (3 * interquartile_range)

    outliers_indices = df_column .> upper_boundary .|| df_column .< lower_boundary
    extreme_outliers_indices = df_column .> extreme_upper_boundary .|| df_column .< extreme_lower_boundary

    return (outliers_indices, extreme_outliers_indices)
end

function getNumberOfOutliers(df, columns)

    outliers = BitArray(undef, size(df, 1))
    outliers .= 0

    extremeOutliers = BitArray(undef, size(df, 1))
    extremeOutliers .= 0

    for column in columns
        column_data = df[!, column]
        
        (outliers_indices, extreme_outliers_indices) = getBoundariesIndices(column_data)

        outliers .= outliers .|| outliers_indices
        extremeOutliers .= extremeOutliers .|| extreme_outliers_indices
    end
    
    numberOfOutliers = sum(outliers)
    numberOfExtremeOutliers = sum(extremeOutliers)

    return numberOfOutliers, numberOfExtremeOutliers
end

function deleteOutliers(df, columns; deleteOnlyExtremeOutliers=false)

    df_copy = copy(df)

    rows_to_delete = BitArray(undef, size(df, 1))
    rows_to_delete .= 0

    for column in columns
        column_data = df[!, column]
        
        (outliers_indices, extreme_outliers_indices) = getBoundariesIndices(column_data)

        if deleteOnlyExtremeOutliers
            rows_to_delete .= rows_to_delete .|| extreme_outliers_indices
        else
            rows_to_delete .= rows_to_delete .|| outliers_indices
        end
    end

    df_copy = df[.!rows_to_delete, :]

    return df_copy
end

function getNumberOfDuplicates(df)

    df_without_duplicates = unique(df)

    numberOfDuplicates = size(df,1) - size(df_without_duplicates,1)

    return numberOfDuplicates
end

function deleteDuplicatedRows(df)

    df_without_duplicates = unique(df)

    return df_without_duplicates
end

function getTotalDepthPercentageErrorIndices(df; error_rate = 0.05)

    df_copy = copy(df)

    df_copy[!,"total_depth_percentage_check"] = 100 .* df[!,:depth] .*2 ./ (df[!,:length] .+ df[!,:width])
    
    real_values = df[!,"total_depth_percentage"]
    aproximated_values = df_copy[!,"total_depth_percentage_check"]

    absolute_error = abs.(real_values .- aproximated_values)

    relative_error = absolute_error ./ abs.(real_values)

    error_rate = 0.05
    
    error_values_indices = relative_error .>= error_rate

    return error_values_indices, df_copy
end

function deleteTotalDepthPercentageErrors(df, error_indices)
    
    df_copy = copy(df)

    df_copy = df[.!error_indices, :]

    return df_copy
end

function addVolumeColumntoDataframe(df)

    df_copy = copy(df)

    df_copy[!,"volume"] = df[!,:depth] .* df[!,:width] .* df[!,:length]

    return df_copy
end

"""
Plots barchart, histogram, scatter, boxplot, heatmap charts of a dataframe.
"""
function plot_dataframe(df, categorical_columns, numerical_columns; save_charts = false)
    
    #= # Bar charts (categorical features).

    plot_name = "barchart"

    for column in categorical_columns

        df_column = combine(groupby(df, [column]), nrow)
    
        number_of_rows = nrow(df)
    
        df_column.percentage = df_column.nrow ./ number_of_rows
    
        current_plot = plot(df_column[!,column], df_column[!,"percentage"], kind = "bar", Layout(xaxis_title= column, yaxis_title = "Relative frequency"))
        if save_charts
            savefig(current_plot, "$(plot_name)_$(column).pdf")
        end
    end

    # Histogram (numerical features).

    plot_name = "histogram"

    for column in numerical_columns
        current_plot = plot(df, x = column, kind = "histogram", Layout(xaxis_title= column, yaxis_title = "Relative frequency"))
        if save_charts
            savefig(current_plot, "$(plot_name)_$(column).png")
        end
    end

    # Scatter plots.

    plot_name = "scatter"

    count1 = 0
    for column_1 in numerical_columns
        count1 += 1
        count2 = 0

        for column_2 in numerical_columns
            count2 += 1

            if count1 >= count2
                continue
            end

            current_plot = plot(df, x = column_1, y = column_2, color = :cut, mode = "markers")
            if save_charts
                savefig(current_plot, "$(plot_name)_$(column_1)_vs_$(column_2).png")
            end
        end
    end

    # Boxplots.

    plot_name = "boxplot"

    for column in numerical_columns
        current_plot = plot(df, x = :cut, y = column, kind = "box")
        if save_charts
            savefig(current_plot, "$(plot_name)_$(column).png")
        end
    end

    # Heatmaps.

    plot_name = "heatmap"

    cols = numerical_columns    # define subset
    M = cor(Matrix(df[!,cols])) # correlation matrix

    (n,m) = size(M)
    current_plot = heatmap(M, fc=cgrad([:white,:dodgerblue4]), xticks=(1:m,cols), xrot=90, yticks=(1:m,cols), yflip=true)
    annotate!([(j, i, text(round(M[i,j],digits=3), 8,"Computer Modern",:black)) for i in 1:n for j in 1:m])
    
    if save_charts
        savefig(current_plot, "$(plot_name).png")
    end =#

    # Scatter 3D plot.

    plot_name = "scatter_3d"

    count1 = 0
    for column_1 in numerical_columns
        count1 += 1
        count2 = 0

        for column_2 in numerical_columns
            count2 += 1
            count3 = 0

            if count1 >= count2
                continue
            end
            
            for column_3 in numerical_columns
                count3 += 1

                if count2 >= count3
                    continue
                end

                current_plot = plot(
                    df, x = column_1, y = column_2, z = column_3, 
                    color = :cut, type = "scatter3d", mode = "markers", marker_size = 0.5
                )

                if save_charts
                    savefig(current_plot, "$(plot_name)_$(column_1)_vs_$(column_2)_vs_$(column_3).png")
                end
            end
        end
    end
end

function get_ordinal_column(df, column, classes)
    column = Vector{String}(df[!, column])
    return ordinalEncoding(column, classes)
end

function diamondsOrdinalEncoding(df)

    df_copy = copy(df)

    GIA_color_grading_scale_ordered = ["J", "I", "H", "G","F", "E", "D"] # Diamonds color grading scale standarized by GIA.
    GIA_clarity_grading_scale_ordered = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"] # GIA diamond clarity grading scale.
    
    df_copy[!, :color]    = get_ordinal_column(df, :color,     GIA_color_grading_scale_ordered)
    df_copy[!, :clarity]  = get_ordinal_column(df, :clarity,   GIA_clarity_grading_scale_ordered)

    return df_copy
end

function diamondTargetOneHotEncoding(df)

    cut_range_ordered = ["Fair", "Good", "Very Good", "Premium", "Ideal"] # cut ordered from worst to best.

    cut_column = df[!, :cut]
    cut_encoded = oneHotEncoding(cut_column, cut_range_ordered)

    return cut_encoded
end

function getZeroMeanNormalizationParameters(df, columns)

    columns_matrix = Matrix{Float64}(df[!, columns])

    normalization_parameters = calculateZeroMeanNormalizationParameters(columns_matrix) # Only with train set.

    return normalization_parameters
end

function getZeroMeanNormalizedColumns(df, columns, normalization_parameters)

    columns_matrix = Matrix{Float64}(df[!, columns])

    columns_zeroMean = normalizeZeroMean(columns_matrix, normalization_parameters)

    return columns_zeroMean
end

function getMinMaxNormalizationParameters(df, columns)

    columns_matrix = Matrix{Float64}(df[!, columns])

    normalization_parameters = calculateMinMaxNormalizationParameters(columns_matrix)

    return normalization_parameters
end

function getMinMaxNormalizedColumns(df, columns, normalization_parameters)

    columns_matrix = Matrix{Float64}(df[!, columns])

    columns_minMax = normalizeMinMax(columns_matrix, normalization_parameters)

    return columns_minMax
end

function performCrossValidationTests(parameters, common_parameters, model, inputs, targets, kfoldindex)

    results_matrix = Array{Any,1}()

    table_headers = Array{String, 1}(undef, length(parameters))
    for i = 1:length(parameters)

        current_parameter = convert(Dict{String, Any}, parameters[i])

        header = ""
        for (key, value) in current_parameter
            header *= key * "=" * string(value) * ", "
        end
        header = header[1:end-2]
        table_headers[i] = header

        for (key, value) in common_parameters
            current_parameter[key] = value
        end

        push!(results_matrix, modelCrossValidation(model, current_parameter, inputs, targets, kfoldindex))
    end

    println(results_matrix)

    pretty_table(header=[model,"Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Fscore"],hcat(
        table_headers,
        [[round(results_matrix[i][1][j], digits=4) for i in 1:length(results_matrix)] for j in 1:6]...
        )  
    )

end