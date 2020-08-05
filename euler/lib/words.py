class Words(object):
    def __init__(self):
        # https://www.ef.com/wwen/english-resources/english-grammar/numbers-english/
        x = '''1	one	first
        2	two	second
        3	three	third
        4	four	fourth
        5	five	fifth
        6	six	sixth
        7	seven	seventh
        8	eight	eighth
        9	nine	ninth
        10	ten	tenth
        11	eleven	eleventh
        12	twelve	twelfth
        13	thirteen	thirteenth
        14	fourteen	fourteenth
        15	fifteen	fifteenth
        16	sixteen	sixteenth
        17	seventeen	seventeenth
        18	eighteen	eighteenth
        19	nineteen	nineteenth
        20	twenty	twentieth
        30	thirty	thirtieth
        40	forty	fortieth
        50	fifty	fiftieth
        60	sixty	sixtieth
        70	seventy	seventieth
        80	eighty	eightieth
        90	ninety	ninetieth
        100	hundred	hundredth'''

        x = x.split('\n')
        x = [_.split()[:2] for _ in x]
        self.table = {
            int(_[0]): _[1]
            for _ in x
        }

    def words(self, k: int):
        if k == 1000:
            return 'one thousand'

        x = self.table
        if k <= 20:
            return x[k]

        if k < 100:
            a, b = k // 10, k % 10
            if b == 0:
                return x[a * 10]
            return '{}-{}'.format(x[a * 10], self.words(b))

        a, b = k // 100, k % 100
        if b == 0:
            return '{} hundred'.format(x[a])
        return '{} hundred and {}'.format(x[a], self.words(b))
