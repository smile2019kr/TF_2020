from dataclasses import dataclass

@dataclass
class Entity:
    context: str
    fname: str  #읽어들일 파일이름
    train: object  #분석대상. 텐서로 전환되므로 객체로 정의.
    test: object  #분석대상. 텐서로 전환되므로 객체로 정의. feature 보다 한단계 상위. 시트.
    id: str #id를 활용해서 분석 전반에서 활용하므로 init - service까지 적용가능하도록 미리설정해줌
    label: str

    @property
    def context(self) -> str: return self._context

    @property
    def fname(self) -> str: return self._fname

    @property
    def train(self) -> object: return self._train

    @property
    def test(self) -> object: return self._test

    @property
    def id(self) -> object: return self._id

    @property
    def label(self) -> str: return self._label


    @context.setter
    def context(self, context): self._context = context

    @fname.setter
    def fname(self, fname): self._fname = fname

    @train.setter
    def train(self, train): self._train = train

    @test.setter
    def test(self, test): self._test = test

    @id.setter
    def id(self, id): self._id = id

    @label.setter
    def label(self, label): self._label = label
