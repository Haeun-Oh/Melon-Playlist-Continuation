# Melon-Playlist-Continuation
[Kakao arena](https://arena.kakao.com/c/8)

### Competition Objective
주어진 플레이리스트의 정보를 참조하여 해당 플레이리스트에 원래 수록되어 있었을 곡 리스트와 태그 리스트를 예측하기
각 플레이리스트별로 원래 플레이리스트에 들어있었을 것이라 예상되는 곡 100개, 태그 10개를 제출

### Grading Method
점수는 예측 곡과 예측 태그의 nDCG의 가중평균값으로 계산된다.
score = 평균 nDCG(예측한 곡) * 0.85 + 평균 nDCG(예측한 태그) * 0.15

### Dataset
* 플레이리스트 메타데이터
  * 플레이리스트 제목
  * 플레이리스트에 수록된 곡
  * 플레이리스트에 달려있는 태그 목록
  * 플레이리스트 좋아요 수
  * 플레이리스트가 최종 수정된 시각
* 곡 메타데이터
  * 곡 제목
  * 앨범 제목
  * 아티스트명
  * 장르
  * 발매일

### Model
use user-based learning

참고 논문: [Efficient K-NN for Playlist Continuation](https://eprints.sztaki.hu/9560/1/Kelen_1_30347064_ny.pdf)
a light-weight playlist-based nearest neighbor method로 간단하지만 강력하고 계산 결과를 빠르게 볼 수 있다는 장점이 있어서 본 논문을 참고했으며, 본 대회의 채점 방식인 nDCG를 위해 solution을 최적화했다는 점에서 이 대회의 문제를 해결하기에 적합한 논문이라고 판단했다.
