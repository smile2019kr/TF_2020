#class Entity:
#    def __init__(self): #py3.8부터는 2,3행이 삭제되고 context: str로 간소화됨. 지원은 됨. 코딩만 간소화된 것
#        pass
from dataclasses import dataclass


@dataclass
class Entity: # 텍스트마이닝 모델은 아래의 3행이 있어야 함
    context: str #_context로 언더바가 없지만 컴파일되면 만들어놓은 init으로 전환되는 것.
    fname: str  #읽어들일 파일이름
    target: str  #분석대상

    @property
    def context(self) -> str: return self._context

    @property
    def fname(self) -> str: return self._fname

    @property
    def target(self) -> str: return self._target

    @context.setter
    def context(self, context): self._context = context

    @fname.setter
    def fname(self, fname): self._fname = fname

    @target.setter
    def target(self, target): self._target = target

