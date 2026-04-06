import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/rpm.js
var headerSeparator, headerLine, simpleEmail, rpmChanges, arch, preamble, section, control_flow_complex, control_flow_simple, operators, rpmSpec;
var init_rpm = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/rpm.js"() {
    headerSeparator = /^-+$/;
    headerLine = /^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)  ?\d{1,2} \d{2}:\d{2}(:\d{2})? [A-Z]{3,4} \d{4} - /;
    simpleEmail = /^[\w+.-]+@[\w.-]+/;
    rpmChanges = {
      name: "rpmchanges",
      token: function(stream) {
        if (stream.sol()) {
          if (stream.match(headerSeparator)) {
            return "tag";
          }
          if (stream.match(headerLine)) {
            return "tag";
          }
        }
        if (stream.match(simpleEmail)) {
          return "string";
        }
        stream.next();
        return null;
      }
    };
    arch = /^(i386|i586|i686|x86_64|ppc64le|ppc64|ppc|ia64|s390x|s390|sparc64|sparcv9|sparc|noarch|alphaev6|alpha|hppa|mipsel)/;
    preamble = /^[a-zA-Z0-9()]+:/;
    section = /^%(debug_package|package|description|prep|build|install|files|clean|changelog|preinstall|preun|postinstall|postun|pretrans|posttrans|pre|post|triggerin|triggerun|verifyscript|check|triggerpostun|triggerprein|trigger)/;
    control_flow_complex = /^%(ifnarch|ifarch|if)/;
    control_flow_simple = /^%(else|endif)/;
    operators = /^(\!|\?|\<\=|\<|\>\=|\>|\=\=|\&\&|\|\|)/;
    rpmSpec = {
      name: "rpmspec",
      startState: function() {
        return {
          controlFlow: false,
          macroParameters: false,
          section: false
        };
      },
      token: function(stream, state) {
        var ch = stream.peek();
        if (ch == "#") {
          stream.skipToEnd();
          return "comment";
        }
        if (stream.sol()) {
          if (stream.match(preamble)) {
            return "header";
          }
          if (stream.match(section)) {
            return "atom";
          }
        }
        if (stream.match(/^\$\w+/)) {
          return "def";
        }
        if (stream.match(/^\$\{\w+\}/)) {
          return "def";
        }
        if (stream.match(control_flow_simple)) {
          return "keyword";
        }
        if (stream.match(control_flow_complex)) {
          state.controlFlow = true;
          return "keyword";
        }
        if (state.controlFlow) {
          if (stream.match(operators)) {
            return "operator";
          }
          if (stream.match(/^(\d+)/)) {
            return "number";
          }
          if (stream.eol()) {
            state.controlFlow = false;
          }
        }
        if (stream.match(arch)) {
          if (stream.eol()) {
            state.controlFlow = false;
          }
          return "number";
        }
        if (stream.match(/^%[\w]+/)) {
          if (stream.match("(")) {
            state.macroParameters = true;
          }
          return "keyword";
        }
        if (state.macroParameters) {
          if (stream.match(/^\d+/)) {
            return "number";
          }
          if (stream.match(")")) {
            state.macroParameters = false;
            return "keyword";
          }
        }
        if (stream.match(/^%\{\??[\w \-\:\!]+\}/)) {
          if (stream.eol()) {
            state.controlFlow = false;
          }
          return "def";
        }
        stream.next();
        return null;
      }
    };
  }
});
init_rpm();
export {
  rpmChanges,
  rpmSpec
};
//# sourceMappingURL=/build/_shared/rpm-WONXWFBR.js.map
