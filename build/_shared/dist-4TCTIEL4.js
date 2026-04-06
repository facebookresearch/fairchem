import {
  addCommonDirectiveOptions,
  commonDirectiveOptions
} from "/build/_shared/chunk-2GZXDR27.js";
import "/build/_shared/chunk-DOEQHMLC.js";
import "/build/_shared/chunk-ZNG4DSCS.js";
import "/build/_shared/chunk-Q6DHUCUI.js";
import "/build/_shared/chunk-LPSXN4QV.js";
import "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/myst-ext-tabs/dist/index.js
var tabSetDirective = {
  name: "tab-set",
  alias: ["tabSet"],
  options: {
    ...commonDirectiveOptions("tab-set")
  },
  body: {
    type: "myst"
  },
  run(data) {
    const tabSet = {
      type: "tabSet",
      children: data.body || []
    };
    addCommonDirectiveOptions(data, tabSet);
    return [tabSet];
  }
};
var tabItemDirective = {
  name: "tab-item",
  alias: ["tabItem", "tab"],
  // TODO: A transform is necessary for stray `tab`s
  arg: {
    type: String
  },
  options: {
    ...commonDirectiveOptions("tab-item"),
    sync: {
      type: String
    },
    selected: {
      type: Boolean
    }
  },
  body: {
    type: "myst"
  },
  run(data) {
    var _a, _b, _c;
    const tabItem = {
      type: "tabItem",
      title: (_a = data.arg) !== null && _a !== void 0 ? _a : "Tab Title",
      sync: (_b = data.options) === null || _b === void 0 ? void 0 : _b.sync,
      selected: (_c = data.options) === null || _c === void 0 ? void 0 : _c.selected,
      children: data.body || []
    };
    addCommonDirectiveOptions(data, tabItem);
    return [tabItem];
  }
};
var tabDirectives = [tabSetDirective, tabItemDirective];
export {
  tabDirectives,
  tabItemDirective,
  tabSetDirective
};
//# sourceMappingURL=/build/_shared/dist-4TCTIEL4.js.map
