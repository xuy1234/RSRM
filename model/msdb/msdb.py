from collections import defaultdict

from model.config import Config


class MSDB:
    def __init__(self, config_s: Config):
        self.config = config_s

    def process_symbol(self, symbol):
        """
        将symbol表达式使用’+‘’-‘分割成不同的子表达式
        :param symbol: 表达式
        :return: 子表达式集合
        """
        symbols = []
        now = ""
        quote = 0
        for s in symbol:
            if s in '({[': quote += 1
            if s in ')}]': quote -= 1
            if s in '+-' and 0 == quote:
                symbols.append(now)
                now = ""
            if s not in ' ':
                now += s
        symbols.append(now)
        return set(map(lambda x: x[1:] if x.startswith('+') else x, symbols))

    def __symbol_add(self, symbols, symbols_count):
        """
        计算加法模式，即寻找T = A+f(x)中的A
        :param pops: 最好的表达式和reward
        :param ga: 遗传算法
        :return: A
        """

        symbols.sort(key=lambda x: x[1])
        symbols = symbols[:self.config.msdb.msdb_max_used_expr_num]
        st = symbols[0][1]
        sym_dict = defaultdict(int)
        tm = 0
        if abs(st) < 1e-99: return ''
        for idx, (tokens, val, symbol) in enumerate(symbols):
            if st / val > self.config.msdb.msdb_expr_ratio or idx <= 1:
                for s in self.process_symbol(symbol):
                    if not s: continue
                    sym_dict[s] += symbols_count[tokens]
                tm += 1
        syms_now = ""
        for symbol, times in sym_dict.items():
            if times >= tm * self.config.msdb.msdb_token_ratio:
                if not syms_now or symbol.startswith('-'):
                    syms_now += symbol
                else:
                    syms_now += '+' + symbol
        if syms_now: syms_now = syms_now + '+'
        return syms_now

    def __symbol_mul(self, symbols):
        """
        计算乘法模式，即寻找T = A*f(x)中的A
        :param pops: 最好的表达式和reward
        :param ga: 遗传算法
        :return: A
        """
        symbols.sort(key=lambda x: x[1])
        syms_now = ""
        st = symbols[0][1]
        if abs(st) < 1e-99: return ''
        for idx, (tokens, val, symbol) in enumerate(symbols):
            flag = 1
            for sym in self.process_symbol(symbol):
                if not sym.startswith('C*') and not sym == 'C':
                    flag = 0
            if flag:
                syms_now = f"({symbol})"
                break

        if syms_now: syms_now = syms_now + '*'
        return syms_now

    def __symbol_pow(self, symbols):
        """
        计算幂次模式，即寻找T = A**f(x)中的A
        :param pops: 最好的表达式和reward
        :param ga: 遗传算法
        :return: A
        """
        symbols.sort(key=lambda x: x[1])
        syms_now = ""
        st = symbols[0][1]
        if abs(st) < 1e-99: return ''
        for idx, (tokens, val, symbol) in enumerate(symbols):
            flag = 1
            for sym in self.process_symbol(symbol):
                if not sym.count('**') or len(sym) >= 3:
                    flag = 0
            if flag:
                syms_now = f"({symbol.split('**')[0]})"
                break
        if syms_now: syms_now = syms_now + '**'
        return syms_now

    def get_form(self, symbols, symbols_count):
        if ('Mul' in self.config.msdb.form_type) and (form := self.__symbol_mul(symbols)):
            return form
        if ('Pow' in self.config.msdb.form_type) and (form := self.__symbol_pow(symbols)):
            return form
        if ('Add' in self.config.msdb.form_type) and (form := self.__symbol_add(symbols, symbols_count)):
            return form
        return ""
