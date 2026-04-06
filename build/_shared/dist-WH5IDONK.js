import {
  addCommonDirectiveOptions,
  commonDirectiveOptions
} from "/build/_shared/chunk-2GZXDR27.js";
import "/build/_shared/chunk-DOEQHMLC.js";
import "/build/_shared/chunk-ZNG4DSCS.js";
import {
  createId,
  normalizeLabel
} from "/build/_shared/chunk-Q6DHUCUI.js";
import "/build/_shared/chunk-LPSXN4QV.js";
import "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/myst-ext-exercise/dist/exercise.js
var exerciseDirective = {
  name: "exercise",
  alias: ["exercise-start"],
  arg: {
    type: "myst"
  },
  options: {
    ...commonDirectiveOptions("exercise"),
    nonumber: {
      type: Boolean,
      doc: "Legacy flag to disable numbering of exercises; equivalent to `enumerated: false`"
    },
    hidden: {
      type: Boolean
    }
  },
  body: {
    type: "myst"
  },
  run(data) {
    var _a, _b, _c, _d, _e;
    const children = [];
    if (data.arg) {
      children.push({
        type: "admonitionTitle",
        children: data.arg
      });
    }
    if (data.body) {
      children.push(...data.body);
    }
    const exercise = {
      type: "exercise",
      hidden: (_a = data.options) === null || _a === void 0 ? void 0 : _a.hidden,
      children
    };
    addCommonDirectiveOptions(data, exercise);
    if (((_b = data.options) === null || _b === void 0 ? void 0 : _b.nonumber) !== void 0) {
      exercise.enumerated = !data.options.nonumber;
    } else {
      exercise.enumerated = (_d = (_c = data.options) === null || _c === void 0 ? void 0 : _c.enumerated) !== null && _d !== void 0 ? _d : true;
    }
    const backupLabel = exercise.enumerated ? `exercise-${createId()}` : void 0;
    const rawLabel = ((_e = data.options) === null || _e === void 0 ? void 0 : _e.label) || backupLabel;
    const { label, identifier } = normalizeLabel(rawLabel) || {};
    exercise.label = label;
    exercise.identifier = identifier;
    if (data.name.endsWith("-start")) {
      exercise.gate = "start";
    }
    return [exercise];
  }
};
var solutionDirective = {
  name: "solution",
  alias: ["solution-start"],
  arg: {
    type: String,
    required: true
  },
  options: {
    ...commonDirectiveOptions("solution"),
    hidden: {
      type: Boolean
    }
  },
  body: {
    type: "myst"
  },
  run(data) {
    var _a;
    const children = [];
    if (data.arg) {
      const { label, identifier } = normalizeLabel(data.arg) || {};
      children.push({
        type: "admonitionTitle",
        children: [
          { type: "text", value: "Solution to " },
          { type: "crossReference", label, identifier }
        ]
      });
    }
    if (data.body) {
      children.push(...data.body);
    }
    const solution = {
      type: "solution",
      hidden: (_a = data.options) === null || _a === void 0 ? void 0 : _a.hidden,
      children
    };
    addCommonDirectiveOptions(data, solution);
    if (data.name.endsWith("-start")) {
      solution.gate = "start";
    }
    return [solution];
  }
};
var solutionEndDirective = {
  name: "solution-end",
  run: () => [{ type: "solution", gate: "end" }]
};
var exerciseEndDirective = {
  name: "exercise-end",
  run: () => [{ type: "exercise", gate: "end" }]
};
var exerciseDirectives = [
  exerciseDirective,
  exerciseEndDirective,
  solutionDirective,
  solutionEndDirective
];
export {
  exerciseDirective,
  exerciseDirectives,
  solutionDirective
};
//# sourceMappingURL=/build/_shared/dist-WH5IDONK.js.map
