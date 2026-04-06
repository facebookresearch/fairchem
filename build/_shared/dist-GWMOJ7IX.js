import {
  addCommonDirectiveOptions,
  commonDirectiveOptions
} from "/build/_shared/chunk-2GZXDR27.js";
import "/build/_shared/chunk-DOEQHMLC.js";
import "/build/_shared/chunk-ZNG4DSCS.js";
import {
  normalizeLabel
} from "/build/_shared/chunk-Q6DHUCUI.js";
import "/build/_shared/chunk-LPSXN4QV.js";
import "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/myst-ext-grid/dist/index.js
function getColumns(columnString) {
  const columns = (columnString !== null && columnString !== void 0 ? columnString : "1 2 2 3").split(/\s/).map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n)).map((n) => Math.min(Math.max(Math.floor(n), 1), 12));
  if (columns.length === 0 || columns.length > 4)
    return [1, 2, 2, 3];
  return columns;
}
var gridDirective = {
  name: "grid",
  arg: {
    type: String
  },
  // options:
  // // https://sphinx-design.readthedocs.io/en/furo-theme/grids.html#grid-options
  // 'class-container'
  // 'class-row'
  // gutter
  // margin
  // padding
  // reverse
  // outline
  options: {
    ...commonDirectiveOptions("grid")
  },
  body: {
    type: "myst",
    required: true
  },
  run(data) {
    const grid = {
      type: "grid",
      columns: getColumns(data.arg),
      children: data.body
    };
    addCommonDirectiveOptions(data, grid);
    return [grid];
  }
};
var gridItemDirective = {
  name: "grid-item",
  options: {
    label: {
      type: String,
      alias: ["name"]
    },
    class: {
      type: String
    }
  },
  body: {
    type: "myst",
    required: true
  },
  run(data) {
    var _a, _b;
    const { label, identifier } = normalizeLabel((_a = data.options) === null || _a === void 0 ? void 0 : _a.label) || {};
    const div = {
      type: "div",
      children: data.body,
      class: (_b = data.options) === null || _b === void 0 ? void 0 : _b.class,
      label,
      identifier
    };
    return [div];
  }
};
var gridDirectives = [gridDirective, gridItemDirective];
export {
  gridDirective,
  gridDirectives,
  gridItemDirective
};
//# sourceMappingURL=/build/_shared/dist-GWMOJ7IX.js.map
