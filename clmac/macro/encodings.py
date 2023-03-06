"""
a collection of all my available macro
"""

from collections import Counter
from functools import partial
from typing import Callable
from typing import Tuple

import pandas as pd
from pynput.keyboard import KeyCode

from clmac.config import boilerplate, conftk
from clmac.helpers.typer import Typer
from clmac.macro.macros import clipper, formatters, writers, launchers, img2text, text2speech
from clmac.macro.macros.launchers import clipboard2browser


class MacroEncoding:
    """
    Represents a specific macro including its name, callable functionality and the keyboard encoding that triggers it
    """

    def __init__(self, cat: str, name: str, encoding: str, func: Callable):
        self.category = cat
        self.name = name
        self.encoding = encoding
        self.func = func
        self.encode_set = self.get_encoding_set()

    def get_encoding_set(self) -> Tuple:
        """get a pynputs representation of the keyboard encoding"""
        return tuple(KeyCode(char=char) for char in self.encoding)

    def __repr__(self):
        return f'MacroEncoded(name="{self.name}", encoding="{self.encoding}", func="{self.func}")'

    def get_set_func_pair(self) -> Tuple[Tuple, Callable]:
        """return a tuple containing the pynputs encoding set and the callable functionality"""
        return self.encode_set, self.func

    def get_text_properties(self) -> Tuple[str, str, str]:
        """return a tuple of the macro string properties"""
        return self.category, self.name, self.encoding


def load_and_type(setting: str) -> Callable:
    """load a setting from the config file and pass into a function (that will type out the setting) that is returned"""

    def type_detail():
        settings = conftk.load_personal()
        try:
            typer.type_text(settings[setting].replace('\\n', '\n'))
        except TypeError:
            print(f'No {setting} found, set config with $ mcli config set -a')

    return type_detail


def load_and_type_numkey(num: int, settings_loader: Callable) -> Callable:
    def type_detail():
        settings = settings_loader()
        try:
            text = settings[num].strip().replace('\\n', '\n')
            typer.type_text(text)
        except TypeError:
            print(f'No {num} found')

    return type_detail


typer = Typer()

load_and_type_numkey_0 = partial(load_and_type_numkey, settings_loader=conftk.load_numkeys_0)
load_and_type_numkey_1 = partial(load_and_type_numkey, settings_loader=conftk.load_numkeys_1)

ENCODINGS = [
    # personal info
    MacroEncoding(cat='personal', name='hotmail', encoding=';hm', func=load_and_type('hotmail')),
    MacroEncoding(cat='personal', name='gmail', encoding=';gm', func=load_and_type('gmail')),
    MacroEncoding(cat='personal', name='work_email', encoding=';wm', func=load_and_type('work_mail')),
    MacroEncoding(cat='personal', name='name', encoding=';al', func=load_and_type('name')),
    MacroEncoding(cat='personal', name='mobile', encoding=';mb', func=load_and_type('mobile')),
    MacroEncoding(cat='personal', name='username', encoding=';un', func=load_and_type('username')),
    MacroEncoding(cat='personal', name='address', encoding=';ad', func=load_and_type('address')),
    # email writer
    MacroEncoding(cat='writer', name='Thanks for you email', encoding=';tf', func=typer('Thanks for your email. ')),
    MacroEncoding(cat='writer', name='Any help would be', encoding=';ah', func=typer(boilerplate.any_help)),
    MacroEncoding(cat='writer', name='Please let me know queries', encoding=';pl', func=typer(boilerplate.please_queries)),
    MacroEncoding(cat='writer', name='Please find attached ', encoding=';pf', func=typer('Please find attached the ')),
    MacroEncoding(cat='writer', name='best_alex', encoding=';ca', func=typer('\nBest\nAlex')),
    MacroEncoding(cat='writer', name='many_thanks_alex', encoding=';mt', func=typer('\n\nMany thanks\n\nAlex')),
    # typing scripts
    MacroEncoding(cat='writer', name='1mil', encoding=';om', func=typer(' / 1000000')),
    MacroEncoding(cat='writer', name='reset_index', encoding=';ri', func=typer('.reset_index()')),
    MacroEncoding(cat='writer', name='set_index', encoding=';si', func=typer('.set_index()', 1)),
    MacroEncoding(cat='writer', name='start_time', encoding=';st', func=typer('start = time.time()\nprint(str(datetime.now()))')),
    MacroEncoding(cat='writer', name='end_time', encoding=';et', func=typer('print(slibtk.hr_secs(time.time() - start))')),
    MacroEncoding(cat='writer', name='train_test_split', encoding=';tt', func=typer(boilerplate.train_test_split)),
    MacroEncoding(cat='writer', name='conda activate', encoding=';co', func=typer('conda activate ')),
    MacroEncoding(cat='writer', name='conda deactivate', encoding=';cd', func=typer('conda deactivate')),
    MacroEncoding(cat='writer', name='.head()', encoding=';;h', func=typer('.head(15)')),
    MacroEncoding(cat='writer', name='.dtype()', encoding=';;d', func=typer('.dtypes')),
    MacroEncoding(cat='writer', name='.shape', encoding=';;s', func=typer('.shape')),
    MacroEncoding(cat='writer', name='ascending', encoding=';as', func=typer('ascending=False')),
    MacroEncoding(cat='writer', name='docker_container', encoding=';dc', func=typer('docker container ')),
    MacroEncoding(cat='writer', name='docker_image', encoding=';di', func=typer('docker image ')),
    MacroEncoding(cat='writer', name='-> DataFrame', encoding=';td', func=typer(' -> pd.DataFrame:')),
    MacroEncoding(cat='writer', name='DataFrame', encoding=';df', func=typer('DataFrame')),
    MacroEncoding(cat='writer', name='if __name__', encoding=';nm', func=typer("if __name__ == '__main__':\n    ")),
    MacroEncoding(cat='writer', name='__init__', encoding=';ii', func=typer('def __init__')),
    MacroEncoding(cat='writer', name='type_print', encoding=';;;', func=writers.write_print),
    MacroEncoding(cat='writer', name='.columns', encoding=';;c', func=typer('.columns')),
    MacroEncoding(cat='writer', name='logging', encoding=';lg', func=typer(boilerplate.logger)),
    MacroEncoding(cat='writer', name='log_config', encoding=';lc', func=typer(boilerplate.log_config)),
    MacroEncoding(cat='writer', name='super().__init__()', encoding=';su', func=typer('super().__init__()')),
    # MacroEncoding(cat='writer', name='modules_data_sci', encoding=';ds', func=typer(boilerplate.data_sci)),
    MacroEncoding(cat='writer', name='modules_deep_learning', encoding=';dl', func=typer(boilerplate.deep_learning)),
    MacroEncoding(cat='writer', name='iris_dataset', encoding=';is', func=typer(boilerplate.iris_dataset)),
    MacroEncoding(cat='writer', name='months', encoding=';mh', func=typer(boilerplate.months)),
    MacroEncoding(cat='writer', name='months_abbr', encoding=';ma', func=typer(boilerplate.months_abbr)),
    MacroEncoding(cat='writer', name='df_columns', encoding=';cn', func=writers.type_columns),
    MacroEncoding(cat='writer', name='.sort_values()', encoding=';sv', func=writers.type_sort_values),
    MacroEncoding(cat='writer', name='sample_head', encoding=';sh', func=typer('.sample(frac=1.).head(100)')),
    MacroEncoding(cat='writer', name='value_counts', encoding=';vc', func=typer('.value_counts()')),
    MacroEncoding(cat='writer', name='n_pct', encoding=';np', func=typer('COUNT(*) n, COUNT(*) / sum(COUNT(*)) over () pct')),
    MacroEncoding(cat='writer', name='memory_usage', encoding=';mu', func=typer('.info(memory_usage=\'deep\')')),
    MacroEncoding(cat='writer', name=': -> None:', encoding=';tn', func=typer(' -> None:')),
    MacroEncoding(cat='writer', name='select_from', encoding=';sf', func=typer('select * from ')),
    MacroEncoding(cat='writer', name=r'select_\n_from', encoding=';fs', func=typer('select * \nfrom ')),
    MacroEncoding(cat='writer', name='select_from_upper', encoding=';SF', func=typer('select * \nfrom ')),
    MacroEncoding(cat='writer', name='left join', encoding=';lj', func=typer('left join')),
    MacroEncoding(cat='writer', name='limit_20', encoding=';ll', func=typer('.limit(5).toPandas()')),
    MacroEncoding(cat='writer', name='create_or_replace', encoding=';cr', func=typer('create or replace table ')),
    MacroEncoding(cat='writer', name='sql_duplicates', encoding=';sd', func=typer(boilerplate.sql_template_duplicates)),
    MacroEncoding(cat='writer', name='distribution', encoding=';dn', func=typer('distribution')),
    MacroEncoding(cat='writer', name='primary_key', encoding=';pk', func=typer('primary_key=')),
    MacroEncoding(cat='writer', name='auto_open', encoding=';ao', func=typer('AUTO_OPEN: bool = False')),
    MacroEncoding(cat='writer', name='plotly_axis_pct', encoding=';pa', func=typer("""fig.update_layout({'yaxis': {'tickformat': '.0%'}})""")),
    MacroEncoding(cat='writer', name='.query(f'')', encoding=';qy', func=typer(""".query(f'')""", 2)),
    MacroEncoding(cat='writer', name='n_customer', encoding=';nc', func=typer('n_customer')),
    MacroEncoding(cat='writer', name='n_transaction', encoding=';nt', func=typer('n_transaction')),
    MacroEncoding(cat='writer', name='.iloc[:25, :25]', encoding=';il', func=typer('.iloc[:25, :25]')),
    MacroEncoding(cat='writer', name='data scientist', encoding=';dd', func=typer('Data Scientist')),
    MacroEncoding(cat='writer', name='subplots', encoding=';sp', func=typer('fig, ax = plt.subplots()')),
    MacroEncoding(cat='writer', name='rename', encoding=';rn', func=typer('.rename({}, axis=1)', 10)),
    MacroEncoding(cat='writer', name='#%%', encoding=';;n', func=typer('#%%')),
    MacroEncoding(cat='writer', name='python -m src.', encoding=';ps', func=typer('python -m src.')),
    MacroEncoding(cat='writer', name=" suffixes=('', '_DROP')", encoding=';sx', func=typer(" suffixes=('', '_DROP')")),
    MacroEncoding(cat='writer', name="deactivate", encoding=';da', func=typer("deactivate")),
    MacroEncoding(cat='writer', name='minute_dt', encoding=';md', func=typer('minute_dt')),
    MacroEncoding(cat='writer', name='.toPandas()', encoding=';tp', func=typer('.toPandas()')),
    MacroEncoding(cat='writer', name="agg_request", encoding=';ar', func=typer(".agg(fn.count('*').alias('n_request'))")),
    MacroEncoding(cat='writer', name="n_request", encoding=';nr', func=typer("n_requestd")),
    MacroEncoding(cat='writer', name="display_plotly", encoding=';dp', func=typer("display_plotly(fig)")),
    MacroEncoding(cat='writer', name="git_branch", encoding=';gb', func=typer("git branch")),
    MacroEncoding(cat='writer', name="git_status", encoding=';gs', func=typer("git status")),
    MacroEncoding(cat='writer', name="git_checkout", encoding=';gc', func=typer("git checkout")),
    MacroEncoding(cat='writer', name="pip install -U pip", encoding=';pu', func=typer("pip install -U pip")),
    MacroEncoding(cat='writer', name="human-in-the-loop", encoding=';hl', func=typer("human-in-the-loop")),
    # numkey writers 0
    MacroEncoding(cat='numkeys', name='1', encoding=';;1', func=load_and_type_numkey_0(1)),
    MacroEncoding(cat='numkeys', name='2', encoding=';;2', func=load_and_type_numkey_0(2)),
    MacroEncoding(cat='numkeys', name='3', encoding=';;3', func=load_and_type_numkey_0(3)),
    MacroEncoding(cat='numkeys', name='4', encoding=';;4', func=load_and_type_numkey_0(4)),
    MacroEncoding(cat='numkeys', name='5', encoding=';;5', func=load_and_type_numkey_0(5)),
    MacroEncoding(cat='numkeys', name='6', encoding=';;6', func=load_and_type_numkey_0(6)),
    MacroEncoding(cat='numkeys', name='7', encoding=';;7', func=load_and_type_numkey_0(7)),
    MacroEncoding(cat='numkeys', name='8', encoding=';;8', func=load_and_type_numkey_0(8)),
    MacroEncoding(cat='numkeys', name='9', encoding=';;9', func=load_and_type_numkey_0(9)),
    # numkey writers 1
    MacroEncoding(cat='numkeys', name='1', encoding=';11', func=load_and_type_numkey_1(1)),
    MacroEncoding(cat='numkeys', name='2', encoding=';22', func=load_and_type_numkey_1(2)),
    MacroEncoding(cat='numkeys', name='3', encoding=';33', func=load_and_type_numkey_1(3)),
    MacroEncoding(cat='numkeys', name='4', encoding=';44', func=load_and_type_numkey_1(4)),
    MacroEncoding(cat='numkeys', name='5', encoding=';55', func=load_and_type_numkey_1(5)),
    MacroEncoding(cat='numkeys', name='6', encoding=';66', func=load_and_type_numkey_1(6)),
    MacroEncoding(cat='numkeys', name='7', encoding=';77', func=load_and_type_numkey_1(7)),
    MacroEncoding(cat='numkeys', name='8', encoding=';88', func=load_and_type_numkey_1(8)),
    MacroEncoding(cat='numkeys', name='9', encoding=';99', func=load_and_type_numkey_1(9)),
    # run script
    MacroEncoding(cat='app', name='img2text', encoding=';i2', func=img2text.img2text),
    MacroEncoding(cat='app', name='clipboard2browser', encoding=';cb', func=clipboard2browser),
    MacroEncoding(cat='app', name='clipper', encoding=';cl', func=clipper.run),
    MacroEncoding(cat='app', name='caret_to_line_start', encoding=';ls', func=writers.caret_to_line_start),
    MacroEncoding(cat='app', name='caret_to_line_end', encoding=';le', func=writers.caret_to_line_end),
    MacroEncoding(cat='app', name='timestamp', encoding=';ts', func=typer.type_timestamp),
    MacroEncoding(cat='app', name='date', encoding=';de', func=typer.type_date),
    MacroEncoding(cat='app', name='start_guitar', encoding=';gr', func=launchers.guitar_practice),
    MacroEncoding(cat='app', name='text2speech', encoding=';ee', func=text2speech.text2speech),
    MacroEncoding(cat='app', name='morning_sites', encoding=';ms', func=launchers.morning_sites),
    # formatting scripts
    # MacroEncoding(cat='formatter', name='wrap_str', encoding=';;t', func=partial(formatters.wrap_cb, prefix='str')),
    MacroEncoding(cat='formatter', name='-- TEST: ', encoding=';;t', func=typer('-- TEST: ')),
    MacroEncoding(cat='formatter', name='unnest_paraenthesis', encoding=';up', func=formatters.unnest_parathesis),
    MacroEncoding(cat='formatter', name='wrap_print', encoding=';wp', func=partial(formatters.wrap_cb, prefix='print')),
    MacroEncoding(cat='formatter', name='wrap_range', encoding=';wr', func=partial(formatters.wrap_cb, prefix='range')),
    MacroEncoding(cat='formatter', name='wrap_type', encoding=';wt', func=partial(formatters.wrap_cb, prefix='type')),
    MacroEncoding(cat='formatter', name='wrap_sum', encoding=';ws', func=partial(formatters.wrap_cb, prefix='sum')),
    MacroEncoding(cat='formatter', name='wrap_int', encoding=';wi', func=partial(formatters.wrap_cb, prefix='int')),
    MacroEncoding(cat='formatter', name='wrap_join', encoding=';wj', func=partial(formatters.wrap_cb, prefix="' '.join")),
    MacroEncoding(cat='formatter', name='wrap_dict', encoding=';wd', func=partial(formatters.wrap_cb, prefix='dict')),
    MacroEncoding(cat='formatter', name='wrap_list', encoding=';wl', func=partial(formatters.wrap_cb, prefix='list')),
    MacroEncoding(cat='formatter', name='wrap_len', encoding=';;l', func=partial(formatters.wrap_cb, prefix='len')),
    MacroEncoding(cat='formatter', name='wrap_next', encoding=';wn', func=partial(formatters.wrap_cb, prefix='next')),
    MacroEncoding(cat='formatter', name='wrap_next', encoding=';wq', func=partial(formatters.wrap_cb, prefix='tqdm')),
    MacroEncoding(cat='formatter', name='wrap_enumerate', encoding=';we', func=partial(formatters.wrap_cb, prefix='enumerate')),
    MacroEncoding(cat='formatter', name='wrap_fstring', encoding=';ff', func=formatters.wrap_clipboard_in_fstring),
    MacroEncoding(cat='formatter', name='select_line_at_caret', encoding=';sl', func=formatters.line_at_caret_to_cb),
    MacroEncoding(cat='formatter', name='select_word_at_caret', encoding=';sw', func=formatters.word_at_caret_to_cb),
    MacroEncoding(cat='formatter', name='cut_right_equality', encoding=';re', func=formatters.cut_right_equality),
    MacroEncoding(cat='formatter', name='set_equal_to_self', encoding=';se', func=formatters.set_equal_to_self),
    MacroEncoding(cat='formatter', name='fmt_pycharm_params', encoding=';pc', func=formatters.fmt_pycharm_params),
    MacroEncoding(cat='formatter', name='fmt_class_props_multiline', encoding=';cp', func=formatters.fmt_class_properties_multiline),
    MacroEncoding(cat='formatter', name='fmt_class_props_multiassign', encoding=';cm', func=formatters.fmt_class_properties_multiassign),
    MacroEncoding(cat='formatter', name='fmt_repr', encoding=';rr', func=formatters.fmt_repr),
    MacroEncoding(cat='formatter', name='fmt_slug', encoding=';2s', func=formatters.to_snake),
    MacroEncoding(cat='formatter', name='fmt_hash', encoding=';hh', func=formatters.fmt_hash),
    MacroEncoding(cat='formatter', name='fmt_hash', encoding=';hj', func=typer('#' * 120)),
    MacroEncoding(cat='formatter', name='fmt_dash', encoding=';dh', func=formatters.fmt_dash),
    MacroEncoding(cat='formatter', name='fmt_hash_center', encoding=';hc', func=formatters.fmt_hash_center),
    MacroEncoding(cat='formatter', name='fmt_rm_whitespace', encoding=';rw', func=formatters.rm_doublespace),
    MacroEncoding(cat='formatter', name='fmt_rm_blanklines', encoding=';rb', func=formatters.rm_blanklines),
    MacroEncoding(cat='formatter', name='fmt_text_wrap', encoding=';tw', func=formatters.wrap_text),
    MacroEncoding(cat='formatter', name='fmt_list', encoding=';lt', func=formatters.fmt_list),
    MacroEncoding(cat='formatter', name='fmt_seq_as_list', encoding=';tl', func=formatters.to_list),
    MacroEncoding(cat='formatter', name='fmt_as_multiple_lines', encoding=';ml', func=formatters.fmt_as_multiple_lines),
    MacroEncoding(cat='formatter', name='fmt_params_as_multiline', encoding=';pm', func=formatters.fmt_params_as_multiline),
    MacroEncoding(cat='formatter', name='fmt_as_pipe', encoding=';pe', func=formatters.fmt_as_pipe),
    MacroEncoding(cat='formatter', name='fmt_as_one_line', encoding=';ol', func=formatters.fmt_as_one_line),
    MacroEncoding(cat='formatter', name='fmt_underline', encoding=';ul', func=formatters.fmt_underline),
    MacroEncoding(cat='formatter', name='cycle_case', encoding=';cc', func=typer.type_cycled_case),
    MacroEncoding(cat='formatter', name='sql_col_as_mil', encoding=';sm', func=formatters.sql_col_as_mil),
    MacroEncoding(cat='formatter', name='fmt_query', encoding=';fq', func=formatters.fmt_sql_table_as_python),
    MacroEncoding(cat='formatter', name='parse_sql_table', encoding=';pt', func=formatters.parse_sql_table),
    MacroEncoding(cat='formatter', name='parse_sql_table', encoding=';sq', func=formatters.swap_quotation_marks),
    MacroEncoding(cat='formatter', name='camel_to_snake', encoding=';cs', func=formatters.camel_to_snake),
    MacroEncoding(cat='formatter', name='snake_to_camel', encoding=';sc', func=formatters.snake_to_camel),
    MacroEncoding(cat='formatter', name='fmt_variables', encoding=';fv', func=formatters.fmt_print_variables),
    MacroEncoding(cat='formatter', name='clean_whitespace', encoding=';cw', func=formatters.clean_whitespace),
    # select previous lines
    MacroEncoding(cat='line select', name='select_prev_1', encoding=';s1', func=partial(typer.select_previous_lines, n_lines=1)),
    MacroEncoding(cat='line select', name='select_prev_2', encoding=';s2', func=partial(typer.select_previous_lines, n_lines=2)),
    MacroEncoding(cat='line select', name='select_prev_3', encoding=';s3', func=partial(typer.select_previous_lines, n_lines=3)),
    MacroEncoding(cat='line select', name='select_prev_4', encoding=';s4', func=partial(typer.select_previous_lines, n_lines=4)),
    MacroEncoding(cat='line select', name='select_prev_5', encoding=';s5', func=partial(typer.select_previous_lines, n_lines=5)),
    MacroEncoding(cat='line select', name='select_prev_6', encoding=';s6', func=partial(typer.select_previous_lines, n_lines=6)),
    MacroEncoding(cat='line select', name='select_prev_7', encoding=';s7', func=partial(typer.select_previous_lines, n_lines=7)),
    MacroEncoding(cat='line select', name='select_prev_8', encoding=';s8', func=partial(typer.select_previous_lines, n_lines=8)),
    MacroEncoding(cat='line select', name='select_prev_9', encoding=';s9', func=partial(typer.select_previous_lines, n_lines=9)),
]

class DuplicateEncodingError(ValueError):
    pass


def test_for_duplicates() -> None:
    """on program start check for duplicate encodings across the macro that would result in two macro being called
    at once"""
    codes = [macro.encoding for macro in ENCODINGS]
    if len(codes) != len(set(codes)):
        err_msg = f'you have added a duplicate encoding: \n{Counter(codes)}'
        input(err_msg + ', press any key to continue...')
        raise DuplicateEncodingError(err_msg)
