const player = "p";
const otherCar = "b";
const player2 = "g";
const road = "r";

setLegend(
  [
    player,
    bitmap`
2111100000011112
2111000000001112
1111000000001112
2111002222001111
2111002222001111
2110000220000112
1111020000201112
1111022002201112
2111022002201111
2111020000201111
2111000220001112
1111002222001112
1110000000000112
2110000000000111
2110333003330112
2111000000001112`,
  ],
  [
    otherCar,
    bitmap`
2111111111111112
2111333333331112
1111333333331112
2111333333331111
2111322222231111
2111322222231112
1111332222331112
1111322222231112
2111322222231111
2111322222231111
2111322222231112
1111332222331112
1111333333331112
2111333333331111
2111333333331112
2111111111111112`,
  ],
  [
    player2,
    bitmap`
2111177777711112
2111766776671112
1111777777771112
2111772222771111
2111772222771111
2117777227777112
1111727777271112
1111722772271112
2111722772271111
2111727777271111
2111777227771112
1111772222771112
1117777777777112
2117777777777111
2117333773337112
2111777777771112`,
  ],
  [
    road,
    bitmap`
2111111111111112
2111111111111112
1111111111111112
1111111111111111
2111111111111111
2111111111111112
2111111111111112
1111111111111112
1111111111111111
2111111111111111
2111111111111112
2111111111111112
1111111111111112
2111111111111111
2111111111111112
2111111111111112`,
  ]
);

let level = 0;
const levels = [
  map`
rrrrrrr
rrrrrrr
rbrrrbr
rrrrrrr
rrrrrrr
rrrrgrr
rrrprrr`,
];
setBackground("r");

const currentLevel = levels[level];
setMap(currentLevel);

setPushables({
  [player]: [],
});

// START - PLAYER MOVEMENT CONTROLS
function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}

onInput("a", () => {
  if (getFirst(player).x !== 0) {
    getFirst(player).x -= 1;
  }
});

onInput("d", () => {
  if (getFirst(player).x !== 6) {
    getFirst(player).x += 1;
  }
});
onInput("j", () => {
  if (getFirst(player2).x !== 0) {
    getFirst(player2).x -= 1;
  }
});

onInput("l", () => {
  if (getFirst(player2).x !== 6) {
    getFirst(player2).x += 1;
  }
});

// END - PLAYER MOVEMENT CONTROLS
let intervalCount = 0;
let intervalSpeed = 400;
let loop = setInterval(() => {
  intervalCount += 1;
  getAll(otherCar).forEach((car) => {
    console.log(car);
    if (car.y == 6) {
      clearTile(car.x, car.y);
    }
    car.y += 1;
  });
  if (intervalCount % 2 === 0) {
    addSprite(getRandomInt(6), 0, otherCar);
  }
  if (tilesWith(player, otherCar).length !== 0) {
    addText("Game Over!", {
      x: 5,
      y: 6,
      color: color`3`,
    });
    clearInterval(loop);
  }
  if (tilesWith(player2, otherCar).length !== 0) {
    addText("Game Over!", {
      x: 5,
      y: 6,
      color: color`3`,
    });
    clearInterval(loop);
  }
}, intervalSpeed);
