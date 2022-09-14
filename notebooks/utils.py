import dateparser
import spacy.lang.en
import re
import random

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

precision_dict = {
    
    "14": "second",
    "13": "minute",
    "12": "hour",
    "11": "day",
    "10": "month",
    "9": "year",
    "8": "decade",
    "7": "century",
    "6": "millenium",
    "4": "hundred thousand years",
    "3": "million years",
    "0": "billion years"
}

q_regex = re.compile(r'when did (.*) marry (.*)[?]')
wikipedia_url_regex = re.compile(r'http:\/\/(.*)\.wikipedia\.org\/wiki\/(.*)')


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


def parse_date(str_with_date: str, nlp: spacy.lang.en.English, precision):
    
    if precision == '9' or precision.strip().lower() == 'year':
        requires = ['year']
    else:
        requires = ['year', 'month']
    
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
                                     'day': f'{parsed_date.day:0>2}',
                                    'orig_date': sd.text})
    return parsed_dates

def run_model(input_string, tokenizer, model, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)

def extract_node1_node2(question):
    rematches = q_regex.match(question)
    node1 = rematches.group(1)
    node2 = rematches.group(2)
    return node1, node2

def create_other_dates(some_date, num_dates, based_on=None):
    if based_on is None:
        based_on = random.choice(['day', 'month', 'year'])
    year, month, day = some_date.split('-')
    new_dates = []
    while num_dates > 0:
        if based_on == 'day' or based_on == '11':
            new_day = int(day) + num_dates
            if int(new_day) >= 27:
                new_day = 1
            new_dates.append(f"{year}-{month}-{new_day:0>2}")
        elif based_on == 'month' or based_on == '10':
            new_month = int(month) + num_dates
            if new_month > 12:
                new_month = 1
            new_dates.append(f"{year}-{new_month:0>2}-{day}")
        elif based_on == 'year' or based_on == '9':
            new_year = int(year) + num_dates
            if new_year > 2019:
                new_year = 2019
            new_dates.append(f"{new_year}-{month}-{day}")
        num_dates -= 1
    return new_dates

def format_dates(some_date, precision):
    year, month, day = some_date.split('-')
    if precision == 'day' or precision == '11':
        return f"{day} {month_dict[month]} {year}"
    if precision == 'month' or precision == '10':
        return f"{month_dict[month]} {year}"
    return f"{year}"

def split_wikipedia_url(wikipedia_url):
    re_match = re.match(wikipedia_url_regex, wikipedia_url)
    if re_match:
        return re_match.group(1), re_match.group(2)
    return None
