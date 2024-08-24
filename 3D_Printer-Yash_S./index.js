/*
@title: 3D_Printer
@author: Yash S.
@snapshot: image1.png
*/

const width = 100;
const height = 100;

//Editable values!!
const x = bt.randInRange(25, 65);
const y = bt.randInRange(31, 62);
const rail_width = 1; // Can be 1 or 2!

setDocDimensions(width, height);

const t = new bt.Turtle();

//Base
t.jump([10, 10]);
t.setAngle(90);
t.arc(-90, 10);
t.setAngle(0);
t.goTo([35, 20]);
t.goTo([35, 10]);
t.goTo([10, 10]);
t.jump([65, 20]);
t.goTo([80, 20]);
t.arc(-90, 10);
t.goTo([65, 10]);
t.goTo([65, 20]);
t.jump([57, 15]);

//Center block
const center_block = [
  [
    [35, 20],
    [65, 20],
    [65, 10],
    [35, 10],
  ],
];
drawLines(center_block);

t.arc(360, 2);

//Screen and Knob
const screen = [
  [
    [38, 18],
    [38, 12],
    [53, 12],
    [53, 18],
    [38, 18],
  ],
];
drawLines(screen);

//Y Axis
const left_arm = [
  [
    [20, 20],
    [20, 75],
    [22, 75],
    [22, 20],
    [23, 20],
    [23, 75],
    [25, 75],
    [25, 20],
  ],
];
drawLines(left_arm);

const right_arm = [
  [
    [75, 20],
    [75, 75],
    [77, 75],
    [77, 20],
    [78, 20],
    [78, 75],
    [80, 75],
    [80, 20],
  ],
];
drawLines(right_arm);

const top = [
  [
    [17, 75],
    [17, 80],
    [83, 80],
    [83, 75],
    [17, 75],
  ],
];
drawLines(top);

//Plate
const plate = [
  [
    [40, 20],
    [40, 22],
    [30, 22],
    [30, 24],
    [70, 24],
    [70, 22],
    [60, 22],
    [60, 20],
    [55, 20],
    [55, 22],
    [45, 22],
    [45, 20],
    [40, 20],
    [40, 22],
    [60, 22],
  ],
];
drawLines(plate);

//X Axis
const x_axis_left = [
  [
    [25, y],
    [x, y],
    [x, y + rail_width],
    [25, y + rail_width],
    [25, y + (5 - rail_width)],
    [x, y + (5 - rail_width)],
    [x, y + 5],
    [25, y + 5],
  ],
];
drawLines(x_axis_left);

const x_axis_right = [
  [
    [75, y],
    [x + 10, y],
    [x + 10, y + rail_width],
    [75, y + rail_width],
    [75, y + (5 - rail_width)],
    [x + 10, y + (5 - rail_width)],
    [x + 10, y + 5],
    [75, y + 5],
  ],
];
drawLines(x_axis_right);

//Extruder

t.jump([x, y + 7]);
t.setAngle(90);
t.arc(-90, 1);
t.goTo([x + 9, y + 8]);
t.arc(-90, 1);
t.goTo([x + 10, y - 4]);
t.arc(-90, 1);
t.goTo([x + 1, y - 5]);
t.arc(-90, 1);
t.goTo([x, y + 7]);

let pos = null;

t.jump([x + 1, y + 2]);
t.setAngle(-90);
for (let a = 0; a <= 20; a++) {
  t.arc(18, 4);
  pos = t.pos;
  t.goTo([x + 5, y + 2]);
  t.goTo(pos);
}

t.jump([x + 4, y - 5]);
t.goTo([x + 4, y - 6]);
t.goTo([x + 6, y - 6]);
t.jump([x + 4, y - 6]);
t.goTo([x + 4.75, y - 7]);
t.goTo([x + 5.25, y - 7]);
t.goTo([x + 6, y - 6]);
t.goTo([x + 6, y - 5]);

let wire = [];
t.jump([x + 6, y + 8]);
let turtle = t.pos;
let offset = 0;
for (let i = 1; i <= 150; i++) {
  let curves = [
    bt.nurbs([
      [turtle[0] + offset, turtle[1]],
      [60, 90 - offset],
      [75, 78 - offset],
    ]),
  ];
  bt.join(wire, curves);
  offset = offset - 0.01;
}

function toCoords(coords1, coords2 = 0) {
  let newCoords;
  if (coords2 != 0) {
    newCoords = [];
    newCoords[0] = coords1.split(":");
    newCoords[0][0] = alphabet[newCoords[0][0]];
    newCoords[0][1] = parseInt(newCoords[0][1]) + 1;
    newCoords[1] = coords2.split(":");
    newCoords[1][0] = alphabet[newCoords[1][0]];
    newCoords[1][1] = parseInt(newCoords[1][1]) + 1;
    return [newCoords[0].join(":"), newCoords[1].join(":")];
  } else {
    newCoords = coords1.split(":");
    newCoords[0] = alphabet[newCoords[0]];
    newCoords[1] = parseInt(newCoords[1]) + 1;
    return newCoords.join(":");
  }
}

function addBoat(coords, orientation, length) {
  // Renders and updates virtual map
  let tip, tip2, body;
  let actualOrien =
    orientation == "left"
      ? "right"
      : orientation == "right"
      ? "left"
      : orientation == "up"
      ? "down"
      : "up"; //Pass through the render function
  let lv = orientation == "left" ? "h" : orientation == "right" ? "h" : "v"; //I forgot why i named this as lv so no explaination
  tip = toCoords(coords.join(":"));
  if (length < 4) {
    switch (orientation) {
      case "left":
        tip2 = toCoords([coords[0], coords[1] - length + 1].join(":"));
        body = toCoords(
          [coords[0], coords[1] - 1].join(":"),
          [coords[0], coords[1] - length + 2].join(":")
        ); //Shouldnt exist if 2
        //use body[0] and body[1] later (remidning myself)
        break;
      case "right":
        tip2 = toCoords([coords[0], coords[1] + length - 1].join(":"));
        body = toCoords(
          [coords[0], coords[1] + 1].join(":"),
          [coords[0], coords[1] + length - 2].join(":")
        );
        break;
      case "up":
        tip2 = toCoords([coords[0] - length + 1, coords[1]].join(":"));
        body = toCoords(
          [coords[0] - 1, coords[1]].join(":"),
          [coords[0] - length + 2, coords[1]].join(":")
        );
        break;
      case "down":
        tip2 = toCoords([coords[0] + length - 1, coords[1]].join(":"));
        body = toCoords(
          [coords[0] + 1, coords[1]].join(":"),
          [coords[0] + length - 2, coords[1]].join(":")
        );
        break;
    }
    if (length <= 2) {
      let stat = orientation == "right" || orientation == "down" ? false : true; // I have no idea why this logic works tbh
      renderBoat([tip], [body[0], body[0]], actualOrien[0], lv, stat);
    } else {
      renderBoat([tip, tip2], body, actualOrien[0], lv);
    }
  } else if (length == 4) {
    if (bt.rand() <= 0.5) {
      switch (
        orientation //less cool ship
      ) {
        case "left":
          tip2 = toCoords([coords[0], coords[1] - length + 1].join(":"));
          body = toCoords(
            [coords[0], coords[1] - length + 2].join(":"),
            [coords[0], coords[1] - 1].join(":")
          ); //Shouldnt exist if 2
          //use body[0] and body[1] later (remidning myself)
          break;
        case "right": //only this works
          tip2 = toCoords([coords[0], coords[1] + length - 1].join(":"));
          body = toCoords(
            [coords[0], coords[1] + 1].join(":"),
            [coords[0], coords[1] + length - 2].join(":")
          );
          break;
        case "up":
          tip2 = toCoords([coords[0] - length + 1, coords[1]].join(":"));
          body = toCoords(
            [coords[0] - length + 2, coords[1]].join(":"),
            [coords[0] - 1, coords[1]].join(":")
          );
          break;
        case "down":
          tip2 = toCoords([coords[0] + length - 1, coords[1]].join(":"));
          body = toCoords(
            [coords[0] + 1, coords[1]].join(":"),
            [coords[0] + length - 2, coords[1]].join(":")
          );
          break;
      }
      renderBoat([tip, tip2], body, actualOrien[0], lv);
    } else {
      //Cool ship
      switch (orientation) {
        case "left":
          body = toCoords(
            [coords[0], coords[1] - length + 1].join(":"),
            [coords[0], coords[1] - 1].join(":")
          ); //Shouldnt exist if 2
          //use body[0] and body[1] later (remidning myself)
          break;
        case "right":
          body = toCoords(
            [coords[0], coords[1] + 1].join(":"),
            [coords[0], coords[1] + length - 1].join(":")
          );
          break;
        case "up":
          body = toCoords(
            [coords[0] - length + 1, coords[1]].join(":"),
            [coords[0] - 1, coords[1]].join(":")
          );
          break;
        case "down":
          body = toCoords(
            [coords[0] + 1, coords[1]].join(":"),
            [coords[0] + length - 1, coords[1]].join(":")
          );
          break;
      }
      renderBoat([tip], body, actualOrien[0], lv);
    }
  } else {
    switch (orientation) {
      case "left":
        body = toCoords(
          [coords[0], coords[1] - length + 1].join(":"),
          [coords[0], coords[1] - 1].join(":")
        ); //Shouldnt exist if 2
        //use body[0] and body[1] later (remidning myself)
        break;
      case "right":
        body = toCoords(
          [coords[0], coords[1] + 1].join(":"),
          [coords[0], coords[1] + length - 1].join(":")
        );
        break;
      case "up":
        body = toCoords(
          [coords[0] - length + 1, coords[1]].join(":"),
          [coords[0] - 1, coords[1]].join(":")
        );
        break;
      case "down":
        body = toCoords(
          [coords[0] + 1, coords[1]].join(":"),
          [coords[0] + length - 1, coords[1]].join(":")
        );
        break;
    }
    renderBoat([tip], body, actualOrien[0], lv);
  }
  switch (orientation) {
    case "left":
      oMap[coords[0]][coords[1]] = 2; //2 is tip, 1 is body, just for better visualisation
      for (let i = 1; i < length; i++) {
        oMap[coords[0]][coords[1] - i] = 1;
      }
      break;
    case "right":
      oMap[coords[0]][coords[1]] = 2; //2 is tip, 1 is body, just for better visualisation
      for (let i = 1; i < length; i++) {
        oMap[coords[0]][coords[1] + i] = 1;
      }
      break;
    case "up":
      oMap[coords[0]][coords[1]] = 2; //2 is tip, 1 is body, just for better visualisation
      for (let i = 1; i < length; i++) {
        oMap[coords[0] - i][coords[1]] = 1;
      }
      break;
    case "down":
      oMap[coords[0]][coords[1]] = 2; //2 is tip, 1 is body, just for better visualisation
      for (let i = 1; i < length; i++) {
        oMap[coords[0] + i][coords[1]] = 1;
      }
      break;
  }
}

function renderBoat(tipp, bodyy = 0, orientation1, orientation2, rd = false) {
  //pass in 0 for body
  tip(tipp[0], orientation1);
  if (tipp[1]) {
    tip(
      tipp[1],
      orientation1 == "l"
        ? "r"
        : orientation1 == "r"
        ? "l"
        : orientation1 == "u"
        ? "d"
        : "u"
    );
  }
  if (bodyy != 0) {
    body(bodyy[0], bodyy[1], orientation2, rd);
  }
}

drawLines(wire);

drawLines(t.lines());
