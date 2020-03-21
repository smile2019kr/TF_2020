class Model:
    def __init__(self):
        self._num1 = 0 # 외부에서 주어지는 값이 num1이므로 이름 충돌우려 -> 언더바를 주고 self.라는 공간에 저장할 수 있도록 0으로 초기화
        self._num2 = 0
        self._opcode = ''
    # 위의 공간에 계속 데이터를 주입하고(getter) 바깥으로 빼낼것(setter) 을 생성

    @property
    def num1(self) -> int:
        return self._num1  # return 이 있으므로 getter. return하는 타입을 int로 지정. 데코레이터.

    @num1.setter
    def num1(self, num1): # setter이므로 int로 return 한다는 구문이 없음.
        self._num1 = num1 # 13행과 14행의 num1은 이름을 동일하게 설정해야 함. property는 언더스코어로 구분

    @property
    def num2(self) -> int:
        return self._num2

    @num2.setter
    def num2(self, num2):
        self._num2 = num2

    @property
    def opcode(self) -> str:
        return self._opcode

    @opcode.setter
    def opcode(self, opcode):
        self._opcode = opcode


