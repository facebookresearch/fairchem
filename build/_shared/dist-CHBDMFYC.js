import {
  addCommonDirectiveOptions,
  commonDirectiveOptions
} from "/build/_shared/chunk-2GZXDR27.js";
import "/build/_shared/chunk-DOEQHMLC.js";
import "/build/_shared/chunk-ZNG4DSCS.js";
import "/build/_shared/chunk-Q6DHUCUI.js";
import "/build/_shared/chunk-LPSXN4QV.js";
import "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/myst-ext-proof/dist/proof.js
var proofDirective = {
  name: "proof",
  alias: [
    "prf:proof",
    "prf:theorem",
    "prf:axiom",
    "prf:lemma",
    "prf:definition",
    "prf:criterion",
    "prf:remark",
    "prf:conjecture",
    "prf:corollary",
    "prf:algorithm",
    "prf:example",
    "prf:property",
    "prf:observation",
    "prf:proposition",
    "prf:assumption"
  ],
  arg: {
    type: "myst"
  },
  options: {
    ...commonDirectiveOptions("proof"),
    nonumber: {
      type: Boolean,
      doc: "Legacy flag to disable numbering of proofs; equivalent to `enumerated: false`"
    }
  },
  body: {
    type: "myst",
    required: true
  },
  run(data) {
    var _a, _b, _c;
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
    let enumerated;
    if (((_a = data.options) === null || _a === void 0 ? void 0 : _a.nonumber) !== void 0) {
      enumerated = !data.options.nonumber;
    } else {
      enumerated = (_c = (_b = data.options) === null || _b === void 0 ? void 0 : _b.enumerated) !== null && _c !== void 0 ? _c : true;
    }
    const proof = {
      type: "proof",
      kind: data.name !== "proof" ? data.name.replace("prf:", "") : void 0,
      enumerated,
      children
    };
    addCommonDirectiveOptions(data, proof);
    return [proof];
  }
};
export {
  proofDirective
};
//# sourceMappingURL=/build/_shared/dist-CHBDMFYC.js.map
