# Melon-Playlist-Continuation
[Kakao arena](https://arena.kakao.com/c/8)

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

a light-weight playlist-based nearest neighbor
method로 간단하지만 강력하고 계산 결과를 빠르게 볼 수 있다는 장점이 있어서 선택했다.
