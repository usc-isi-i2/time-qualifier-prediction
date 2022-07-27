import dateparser
import spacy.lang.en

month_dict = {
    '01': 'January',
    '02': 'February',
    '03': 'March',
    '04': 'April',
    '05': 'May',
    '06': 'June',
    '07': 'July',
    '08': 'August',
    '09': 'September',
    '10': 'October',
    '11': 'November',
    '12': 'December'
    
}


def match_dates_based_on_precision(date1: str, wikidata_precision: str, date2: dict):
    if date2 == "":
        return  False, ""
    year, month, day = date1.split('-')
    if wikidata_precision == '10' or wikidata_precision.strip().lower() == 'month':
        return year == str(date2['year']) and month == str(date2['month']), 'month'
    if wikidata_precision == '9' or wikidata_precision.strip().lower() == 'year':
        return year == str(date2['year']), 'year'
    if wikidata_precision == '11' or wikidata_precision.strip().lower() == 'day':
        return date1 == date2['date'], 'day'
    return date1 == date2['date'], 'exact'


def parse_date(str_with_date: str, nlp: spacy.lang.en.English, requires=['year', 'month']):
    spacy_docs = nlp(str_with_date)
    spacy_dates = [x for x in spacy_docs.ents if x.label_ == 'DATE']
    parsed_dates = []
    if spacy_dates:
        for sd in spacy_dates:
            parsed_date = dateparser.parse(sd.text, settings={'PREFER_DAY_OF_MONTH': 'first', 
                                                              'REQUIRE_PARTS': requires})
            if parsed_date:
                parsed_dates.append({'date': f'{parsed_date.year}-{parsed_date.month:0>2}-{parsed_date.day:0>2}',
                                     'year': parsed_date.year,
                                     'month': f'{parsed_date.month:0>2}',
                                     'day': f'{parsed_date.day:0>2}'})
    return parsed_dates

def run_model(input_string, tokenizer, model, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)
