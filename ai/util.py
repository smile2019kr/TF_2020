class Storage:
    train_X : object # pd로 뽑아낸 csv파일들은 객체가 되므로 클래스내에서 활용가능한 object로 지정
    train_Y : object
    test_X : object
    test_Y : object

    @property
    def train_X(self) -> object: return self._train_X
    @train_X.setter
    def train_X(self, train_X): self._train_X = train_X

    @property
    def train_Y(self) -> object: return self._train_Y
    @train_Y.setter
    def train_Y(self, train_Y): self._train_Y = train_Y

    @property
    def test_X(self) -> object: return self._test_X
    @test_X.setter
    def test_X(self, test_X): self._test_X = test_X

    @property
    def test_Y(self) -> object: return self._test_Y
    @test_Y.setter
    def test_Y(self, test_Y): self._test_Y = test_Y

