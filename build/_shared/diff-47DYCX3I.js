import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/diff.js
var TOKEN_NAMES, diff;
var init_diff = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/diff.js"() {
    TOKEN_NAMES = {
      "+": "inserted",
      "-": "deleted",
      "@": "meta"
    };
    diff = {
      name: "diff",
      token: function(stream) {
        var tw_pos = stream.string.search(/[\t ]+?$/);
        if (!stream.sol() || tw_pos === 0) {
          stream.skipToEnd();
          return ("error " + (TOKEN_NAMES[stream.string.charAt(0)] || "")).replace(/ $/, "");
        }
        var token_name = TOKEN_NAMES[stream.peek()] || stream.skipToEnd();
        if (tw_pos === -1) {
          stream.skipToEnd();
        } else {
          stream.pos = tw_pos;
        }
        return token_name;
      }
    };
  }
});
init_diff();
export {
  diff
};
//# sourceMappingURL=/build/_shared/diff-47DYCX3I.js.map
