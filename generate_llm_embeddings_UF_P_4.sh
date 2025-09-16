
model_type=$1
other_subsets="single"

# Generate LLM embeddings for UltraFeedback dataset
if [ "${other_subsets}" = "ultra_feedback" ]; then
    subsets="helpfulness honesty instruction_following truthfulness"
elif [ "${other_subsets}" = "single" ]; then
    subsets="8 4 2 1"
elif [ "${other_subsets}" = "84" ]; then
    subsets="8 4"
else
    echo "Invalid!"
fi

echo "${subsets}"

survey_size=16

for subset in ${subsets}
do
    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/P_4_survey_${survey_size}/" \
    --data_path "data/UltraFeedback_${other_subsets}_P_4" --data_subset ${subset} --data_split train --model_type ${model_type} \
    --other_subsets ${other_subsets} --with_embeddings True --survey_size $survey_size --num_duplicates 1

    python -m hidden_context.data_utils.add_survey_contexts --output_dir "data/P_4_survey_${survey_size}/" \
    --data_path "data/UltraFeedback_${other_subsets}_P_4" --data_subset ${subset} --data_split test --model_type ${model_type} \
    --other_subsets ${other_subsets} --with_embeddings True --survey_size $survey_size --num_duplicates 1
done

