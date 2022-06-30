import dateparser
import spacy.lang.en


def match_dates_based_on_precision(date1: str, wikidata_precision: str, date2: dict):
    year, month, day = date1.split('-')
    if wikidata_precision == '10':
        return year == str(date2['year']) and month == str(date2['month']), 'month'
    if wikidata_precision == '9':
        return year == str(date2['year']), 'year'
    if wikidata_precision == '11':
        return date1 == date2['date'], 'day'
    return date1 == date2['date'], 'exact'


def parse_date(str_with_date: str, nlp: spacy.lang.en.English):
    spacy_docs = nlp(str_with_date)
    spacy_dates = [x for x in spacy_docs.ents if x.label_ == 'DATE']
    parsed_dates = []
    if spacy_dates:
        for sd in spacy_dates:
            parsed_date = dateparser.parse(sd.text, settings={'PREFER_DAY_OF_MONTH': 'first'})
            if parsed_date:
                parsed_dates.append({'date': f'{parsed_date.year}-{parsed_date.month:0>2}-{parsed_date.day:0>2}',
                                     'year': parsed_date.year,
                                     'month': f'{parsed_date.month:0>2}',
                                     'day': f'{parsed_date.day:0>2}'})
    return parsed_dates
