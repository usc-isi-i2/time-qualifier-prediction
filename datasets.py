from src.fact_verification_dataset import MarriageFactVerificationDataset
from src.tacred_spouse_fact_verification_dataset import TacredSpouseFactVerificationDataset
from src.tacred_spouse_fact_verification_dataset_combined import TacredSpouseFactVerificationCombinedDataset
from src.tacred_spouse_fact_verification_dataset_empty import TacredSpouseFactVerificationEmptyDataset
from src.marriages import MarriagesWikidataDataset


def get_unified_qa_dataset(dataset_name):
    train_file = f'data/unifiedQA/{dataset_name}.json'
    return MarriageFactVerificationDataset(train_file)


def get_tacred_dataset(dataset_name):
    return TacredSpouseFactVerificationDataset('./data/tacred', f'{dataset_name}.txt')


def get_tacred_combined_dataset(dataset_name):
    return TacredSpouseFactVerificationCombinedDataset('./data/tacred', f'{dataset_name}.txt')


def get_tacred_empty_dataset(dataset_name):
    return TacredSpouseFactVerificationEmptyDataset('./data/tacred', f'{dataset_name}.txt')


def get_anchors_dataset(dummy):
    return MarriageFactVerificationDataset()


def get_marriages_dataset(dataset_name):
    train_file = f'data/marriages/{dataset_name}.json'
    return MarriagesWikidataDataset(train_file)


def get_position_held_dataset(dataset_name):
    train_file = f'data/position/{dataset_name}.json'
    return MarriagesWikidataDataset(train_file)


get_dataset_functions = {
    "unifiedQA": get_unified_qa_dataset,
    "tacred": get_tacred_dataset,
    "anchor": get_anchors_dataset,
    "tacred_combined": get_tacred_combined_dataset,
    "tacred_empty": get_tacred_empty_dataset,
    "marriages": get_marriages_dataset,
    "position": get_position_held_dataset
}
